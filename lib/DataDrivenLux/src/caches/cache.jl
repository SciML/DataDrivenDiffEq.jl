struct SearchCache{ALG, PTYPE, O} <: AbstractAlgorithmCache
    alg::ALG
    candidates::AbstractVector{Candidate}
    ages::AbstractVector{Int}
    keeps::AbstractVector{Bool}
    sorting::AbstractVector{Int}
    p::AbstractVector
    dataset::Dataset
    optimiser_state::O
end

function Base.show(io::IO, cache::SearchCache)
    print(io, "SearchCache : $(cache.alg)")
    return
end

function init_cache(x::X where {X <: AbstractDAGSRAlgorithm}, basis::Basis,
                    problem::DataDrivenProblem; kwargs...)
    @unpack n_layers, functions, arities, skip, rng, populationsize, loss = x
    @unpack optimizer, optim_options, loss, observed, distributed, threaded = x
    @unpack use_protected = x
    @unpack optimiser, keep = x

    # Derive the model
    dataset = Dataset(problem)
    TData = eltype(dataset)

    rng_ = Lux.replicate(rng)

    observed = isa(observed, ObservedModel) ? observed : ObservedModel(dataset.y, fixed = true)

    if use_protected
        functions = map(convert_to_safe, functions)
    end

    model = LayeredDAG(length(basis), size(dataset.y, 1), n_layers, arities, functions,
                       skip = skip; kwargs...)
    ps = Lux.initialparameters(rng_, model)
    ps = ComponentVector(ps)

    pdists = ParameterDistributions(basis, TData)

    # Derive the candidates     
    candidates = map(1:populationsize) do i
        candidate = Candidate(model, ps, rng_, basis, dataset, observed = observed,
                              parameterdist = pdists)
        optimize_candidate!(candidate, ps, dataset, optimizer, optim_options)
        update_values!(candidate, ps, dataset)
        candidate
    end


    keeps = zeros(Bool, populationsize)
    ages = zeros(Int, populationsize)
    sorting = zeros(Int, populationsize)

    if isa(keep, Int)
        sortperm!(sorting, candidates, alg = PartialQuickSort(keep), by = loss)
        permute!(candidates, sorting)
        keeps[1:keep] .= true 
    else
        losses = filter(!isnan, map(loss, candidates))
        # TODO Maybe weight by age or loss here
        loss_quantile = quantile(losses, keep)
        keeps .= (losses .<= loss_quantile)
    end

    # Distributed always goes first here
    if distributed
        ptype = __PROCESSUSE(3)
    elseif threaded
        ptype = __PROCESSUSE(2)
    else
        ptype = __PROCESSUSE(1)
    end

    # Setup the optimiser
    if isa(optimiser, Optimisers.AbstractRule)
        optimiser_state = Optimisers.setup(optimiser, ps[:])
    else
        optimiser_state = nothing
    end
    return SearchCache{typeof(x), ptype, typeof(optimiser_state)}(x, candidates, ages, keeps, sorting, ps, dataset, optimiser_state)
end

function update_cache!(cache::SearchCache)
    @unpack keep, loss, optimizer, optim_options = cache.alg

    # Update the parameters based on the current results
    update_parameters!(cache)

    optimize_cache!(cache, cache.p)

    cache.keeps .= false

    if isa(keep, Int)
        sortperm!(cache.sorting, cache.candidates, alg = PartialQuickSort(keep), by = loss)
        permute!(cache.candidates, cache.sorting)
        cache.keeps[1:keep] .= true 
    else
        losses = filter(!isnan, map(loss, cache.candidates))
        # TODO Maybe weight by age or loss here
        loss_quantile = quantile(losses, keep)
        cache.keeps .= (losses .<= loss_quantile)
    end

    return
end

# Optimizes the cache and returns the loglikelihoods

# Serial 
function optimize_cache!(cache::SearchCache{<:Any, __PROCESSUSE(1)}, p = cache.p)
    @unpack optimizer, optim_options = cache.alg
    map(enumerate(cache.candidates)) do (i, candidate)
        if cache.keeps[i]
            cache.ages[i] += 1
            return true
        else
            optimize_candidate!(candidate, p, cache.dataset, optimizer,
                                optim_options)
            cache.ages[i] = 0
            return true
        end
    end
    return
end

# Threaded
function optimize_cache!(cache::SearchCache{<:Any, __PROCESSUSE(2)}, p = cache.p)
    @unpack optimizer, optim_options = cache.alg

    # Update all 
    Threads.@threads for i in 1:length(cache.keeps)
        if cache.keeps[i]
            cache.ages[i] += 1
        else
            optimize_candidate!(cache.candidates[i], p, cache.dataset, optimizer,
                                optim_options)
            cache.ages[i] = 0
        end
    end
    return
end

# Distributed

function optimize_cache!(cache::SearchCache{<:Any, __PROCESSUSE(3)}, p = cache.p)
    @unpack optimizer, optim_options = cache.alg

    successes = pmap(1:length(cache.keeps)) do i
        if cache.keeps[i]
            cache.ages[i] += 1
            return true
        else
            optimize_candidate!(cache.candidates[i], p, cache.dataset, optimizer,
                                optim_options)
            cache.ages[i] = 0
            return true
        end
    end
    return
end