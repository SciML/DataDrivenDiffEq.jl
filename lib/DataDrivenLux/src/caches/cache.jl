struct SearchCache{ALG, N, M, P, B <: AbstractBasis} <: AbstractAlgorithmCache
    alg::ALG
    candidates::AbstractVector
    ages::AbstractVector{Int}
    sorting::AbstractVector{Int}
    keeps::AbstractVector{Bool}
    paths::AbstractVector{PathState}
    iterations::Int
    model::M
    p::P
    basis::B
    dataset::Dataset
end

function Base.show(io::IO, cache::SearchCache)
    print(io, cache.alg)
    print(io, map(cache.alg.loss,cache.candidates[cache.keeps]))
    return
end

function SearchCache(x::X where X <: AbstractDAGSRAlgorithm, basis::Basis, X::AbstractMatrix, Y::AbstractMatrix, 
        U::AbstractMatrix = Array{eltype(X)}(undef, 0, 0), t::AbstractVector = Array{eltype(X)}(undef, 0); kwargs...)

        @unpack n_layers, functions, arities, skip, rng, populationsize = x
        @unpack optimizer, optim_options, loss, observed= x
        @unpack use_protected = x

        observed = isa(observed, ObservedModel) ? observed : ObservedModel(size(Y,1))
        
        if use_protected
            functions = map(convert_to_safe, functions)
            @info functions
            @info arities
        end

        # Derive the model
        dataset = Dataset(X, Y, U, t)

        model = LayeredDAG(length(basis), size(dataset.y, 1), n_layers, arities, functions, skip = skip; kwargs...)
        ps, st = Lux.setup(rng, model)
        input_paths = [PathState{Float32}(zero(eltype(X)), (), (i,)) for i in 1:length(basis)]
        pdists = ParameterDistributions(basis)
        
        # Derive the candidates     
        candidates = map(1:populationsize) do i
            Candidate(model, ps, st, basis, dataset, observed = observed, parameterdist = pdists)
        end
    
        keeps = zeros(Bool, populationsize)
        ages = zeros(Int, populationsize)
        sorting = sortperm(candidates, by = loss)
        
        return SearchCache{typeof(x), populationsize, typeof(model), typeof(ps), typeof(basis)}(x, candidates, ages, sorting,keeps, input_paths, 0,model, ps, basis, dataset)
    end
    
function update_cache!(cache::SearchCache)
    @unpack candidates, p, model, alg, keeps, sorting, ages, iterations = cache
    @unpack keep, loss, distributed, optimizer, optim_options = alg
    @unpack basis, dataset = cache

    sortperm!(sorting, candidates, by = loss)
    permute!(candidates, sorting)

    keeps .= false

    if isa(keep, Int)
        keeps[1:keep] .= true
    else
        losses = map(loss, candidates)
        # TODO Maybe weight by age or loss here
        loss_quantile = quantile(losses, keep)
        keeps .= losses .<= loss_quantile
    end

    ages[keeps] .+= 1
    ages[.!keeps] .= 0

    # Update the parameters based on the current results
    # Here
    p = update_parameters(alg, p, candidates[keeps])
    
    cache = optimize_cache(cache, p)

    return cache
end

# Optimizes the cache and returns the loglikelihoods
function optimize_cache(cache::SearchCache, p = cache.p)
    @unpack candidates, alg, keeps, dataset = cache
    @unpack distributed, optimizer, optim_options = alg

    # Update all 
    if distributed
        keeps .=  pmap(1:length(keeps)) do i 
            if keeps[i]
                return 
            end
            try
                candidates[i] = sample!(candidates[i], p)
                candidates[i] = optimize_candidate!(candidates[i], p, dataset, optimizer, optim_options)
                return 
            catch e
                @debug "Failed to update candidate $i on $(Distributed.myid())"
                return 
            end
        end
    else
        map(1:length(keeps)) do i 
            if keeps[i]
                return 
            end
            try
                candidates[i] = sample!(candidates[i], p)
                candidates[i] = optimize_candidate!(candidates[i], p, dataset, optimizer, optim_options)
                return 
            catch e
                @debug "Failed to update candidate $i"
                return 
            end
        end
    end 

    return cache
end
