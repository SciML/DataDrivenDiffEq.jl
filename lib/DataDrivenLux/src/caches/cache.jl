struct SearchCache{ALG} <: AbstractAlgorithmCache
    alg::ALG
    candidates::AbstractVector{Candidate}
    ages::AbstractVector{Int}
    keeps::AbstractVector{Bool}
    iterations::Int
    p::AbstractVector
    dataset::Dataset
end

function Base.show(io::IO, cache::SearchCache)
    print(io, cache.alg)
    losses = map(cache.alg.loss, cache.candidates[cache.keeps])
    print(io, DataDrivenDiffEq.StatsBase.summarystats(losses))
    return
end

function init_cache(x::X where {X <: AbstractDAGSRAlgorithm}, basis::Basis,
                    problem::DataDrivenProblem; kwargs...)
    @unpack n_layers, functions, arities, skip, rng, populationsize, loss = x
    @unpack optimizer, optim_options, loss, observed = x
    @unpack use_protected = x

    # Derive the model
    dataset = Dataset(problem)
    TData = eltype(dataset)

    observed = isa(observed, ObservedModel) ? observed : ObservedModel(size(dataset.y, 1))

    if use_protected
        functions = map(convert_to_safe, functions)
    end

    model = LayeredDAG(length(basis), size(dataset.y, 1), n_layers, arities, functions,
                       skip = skip; kwargs...)
    ps = Lux.initialparameters(rng, model)
    ps = ComponentVector(ps)

    pdists = ParameterDistributions(basis, TData)

    # Derive the candidates     
    candidates = map(1:populationsize) do i
        candidate = Candidate(model, ps, rng, basis, dataset, observed = observed,
                              parameterdist = pdists)
        optimize_candidate!(candidate, ps, dataset, optimizer, optim_options)
        update_values!(candidate, ps, dataset)
        candidate
    end

    keeps = zeros(Bool, populationsize)
    ages = zeros(Int, populationsize)

    return SearchCache{typeof(x)}(x, candidates, ages, keeps, 0, ps, dataset)
end

function update_cache!(cache::SearchCache)
    @unpack keep, loss, distributed, optimizer, optim_options = cache.alg

    sort!(cache.candidates, by = loss)

    cache.keeps .= false

    if keep .>= 1
        cache.keeps[1:keep] .= true
    else
        losses = map(loss, cache.candidates)
        # TODO Maybe weight by age or loss here
        loss_quantile = quantile(losses, keep)
        cache.keeps .= (losses .<= loss_quantile)
    end
    # Update the parameters based on the current results
    # Here

    cache.p .= update_parameters(cache.alg, cache.p, cache.candidates[cache.keeps])

    optimize_cache!(cache, cache.p)

    sort!(cache.candidates, by = loss)
    return
end

# Optimizes the cache and returns the loglikelihoods
function optimize_cache!(cache::SearchCache, p = cache.p)
    @unpack distributed, optimizer, optim_options = cache.alg

    # Update all 
    if distributed
        successes = pmap(1:length(cache.keeps)) do i
            if cache.keeps[i]
                cache.ages[i] += 1
                return true
            else
                try
                    optimize_candidate!(cache.candidates[i], p, cache.dataset, optimizer,
                                        optim_options)
                    cache.ages[i] = 0
                    return true
                catch e
                    @debug "Failed to update candidate $i on $(Distributed.myid())"
                    return false
                end
            end
        end
    else
        map(enumerate(cache.candidates)) do (i, candidate)
            if cache.keeps[i]
                cache.ages[i] += 1
                return true
            else
                try
                    optimize_candidate!(candidate, p, cache.dataset, optimizer,
                                        optim_options)
                    cache.ages[i] = 0
                    return true
                catch e
                    @info "Failed to update candidate $i"
                    return false
                end
            end
        end
    end
    return
end
