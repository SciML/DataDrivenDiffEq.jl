
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
    abstol = eps(eltype(T)), progress::Bool = false) where T <: Number

    λ = get_threshold(opt)

    if progress
        progress =  init_progress(opt, X, A, Y, maxiter)
    else
        progress = nothing
    end

    @views opt(X, A, Y, λ, maxiter = maxiter, abstol = abstol, progress = progress)

    return
end

function sparse_regression!(X, A, Y, opt::AbstractOptimizer{T};
    maxiter::Int = maximum(size(A)),
    abstol = eps(eltype(T)), progress::Bool = false,
    f::Function = F(opt),
    g::Function = G(opt)) where T <: AbstractVector

    # Closure for the pareto function
    fg(x, A, y) = (g∘f)(x, A, y)
    #return fg

    # TODO Tmp Result
    X_tmp = deepcopy(X)

    λ = get_threshold(opt)

    if progress
        progress =  init_progress(opt, X, A, Y, maxiter)
    else
        progress = nothing
    end

    for (i,λi) in enumerate(λ)
        @views opt(X_tmp, A, Y, λi, maxiter = maxiter, abstol = abstol, progress = progress)
        for j in 1:size(Y, 2)
            if evaluate_pareto!(view(X, :, j), view(X_tmp, :, j), fg, view(A, :, :), view(Y, :, j))
                clip_by_threshold!(X[:, j], λi)
            end
        end
    end
    return
end
