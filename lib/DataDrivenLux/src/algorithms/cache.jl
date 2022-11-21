abstract type AbstractAlgorithmCache end

struct SearchCache{ALG, N, M, P} <: AbstractAlgorithmCache
    alg::ALG
    candidates::AbstractVector
    ages::AbstractVector{Int}
    sorting::AbstractVector{Int}
    keeps::AbstractVector{Bool}
    iterations::Int
    model::M
    p::P
end

function Base.show(io::IO, cache::SearchCache)
    print("Algorithm :")
    summary(io, cache.alg)
    println(io, "")
    for c in cache.candidates[cache.keeps]
        println(io, c)
    end
    return
end

function SearchCache(x::AbstractDAGSRAlgorithm, X::AbstractMatrix, Y::AbstractMatrix)
    init_cache(x, X, Y)
end

function update!(cache::SearchCache{<:AbstractDAGSRAlgorithm}, X::AbstractMatrix,
                 Y::AbstractMatrix)
    @unpack candidates, p, model, alg, keeps, sorting, ages, iterations = cache
    @unpack keep, loss, procs, optimizer, optim_options = alg

    sortperm!(sorting, candidates, by = loss)

    keeps .= false

    permute!(candidates, sorting)

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
    p = update(p, model, alg, candidates, keeps, X, Y)

    # Update all 
    if isnothing(procs)
        @inbounds for (i, keepidx) in enumerate(keeps)
            if !keepidx
                candidates[i] = resample!(candidates[i], model, p, X, Y, optimizer,
                                          optim_options)
            end
        end
    else
        f_update(c) = resample!(c, model, p, X, Y, optimizer, optim_options)
        candidates[.!keeps] .= pmap(f_update, procs, candidates[.!keeps])
    end

    return
end
