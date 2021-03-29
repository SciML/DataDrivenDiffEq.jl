
"""
$(SIGNATURES)

Implements a sparse regression, given an `AbstractOptimizer` or `AbstractSubspaceOptimizer`.
`maxiter` indicate the maximum iterations for each call of the optimizer, `abstol` the absolute tolerance of
the difference between iterations in the 2 norm. If the optimizer is called with a `Vector` of thresholds, each `maxiter` indicates
the maximum iterations for each threshold.

If `progress` is set to `true`, a progressbar will be available.
"""
function sparse_regression!(X, A, Y, opt::AbstractOptimizer{T};
    maxiter::Int = maximum(size(A)),
    abstol = eps(eltype(T)), progress::Bool = false,
    progress_outer::Int = 1, progress_offset::Int = 0, kwargs...) where T <: Number

    λ = get_threshold(opt)

    if progress
        progress =  init_progress(opt, X, A, Y, maxiter*progress_outer, progress_offset)
    else
        progress = nothing
    end

    opt(X, A, Y, λ, maxiter = maxiter, abstol = abstol, progress = progress)

    return
end

function sparse_regression!(X, A, Y, opt::AbstractOptimizer{T};
    maxiter::Int = maximum(size(A)),
    abstol = eps(eltype(T)), progress::Bool = false,
    f::Function = F(opt),
    g::Function = G(opt),
    progress_outer::Int = 1, progress_offset::Int = 0, kwargs...) where T <: AbstractVector

    # Closure for the pareto function
    fg(x, A, y) = (g∘f)(x, A, y)
    #return fg

    # TODO Tmp Result
    X_tmp = deepcopy(X)

    λ = sort(get_threshold(opt))

    if progress
        progress =  init_progress(opt, X, A, Y, maxiter*length(λ)*progress_outer, progress_offset)
    else
        progress = nothing
    end

    @views for (i,λi) in enumerate(λ)
        init!(X_tmp, opt, A, Y)
        opt(X_tmp, A, Y, λi, maxiter = maxiter, abstol = abstol, progress = progress)
        all(X_tmp .== zero(eltype(X))) && break # Increasing the threshold makes no sense
        for j in 1:size(Y, 2)
            evaluate_pareto!(X[:, j], X_tmp[:, j], fg, A, Y[:,j])
        end
    end
    return
end
