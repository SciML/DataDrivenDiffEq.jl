"""
$(SIGNATURES)

Implements a sparse regression, given an `AbstractOptimizer` or `AbstractSubspaceOptimizer`.
`X` denotes the coefficient matrix, `A` the design matrix and `Y` the matrix of observed or target values.
`X` can be derived via `init(opt, A, Y)`.
`maxiter` indicates the maximum iterations for each call of the optimizer, `abstol` the absolute tolerance of
the difference between iterations in the 2 norm. If the optimizer is called with a `Vector` of thresholds, each `maxiter` indicates
the maximum iterations for each threshold.

If `progress` is set to `true`, a progressbar will be available. `progress_outer` and `progress_offset` are used to compute the initial offset of the
progressbar.

If used with a `Vector` of thresholds, the functions `f` with signature `f(X, A, Y)` and `g` with signature `g(x, threshold) = G(f(X, A, Y))` with the arguments given as stated above can be passed in. These are
used for finding the pareto-optimal solution to the sparse regression. 
"""
@views sparse_regression!(X, A, Y, opt::AbstractOptimizer; kwargs...) = begin 
    init!(X, opt, A, Y) 
    λ_opt =  opt(X, A, Y; kwargs...)
    for i in axes(X, 2)
        clip_by_threshold!(X[:,i], λ_opt[i])
    end
    λ_opt
end
