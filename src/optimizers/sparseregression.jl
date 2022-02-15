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
function sparse_regression!(X, A, Y, opt::AbstractOptimizer{T};
    maxiter::Int = maximum(size(A)),
    abstol = eps(eltype(T)), progress::Bool = false,
    f::Function = F(opt),
    g::Function = G(opt),
    progress_outer::Int = 1, progress_offset::Int = 0, kwargs...) where T

    # Closure for the pareto function
    fg(x, A, y, lambda) = (g∘f)(x, A, y, lambda)
    
    # TODO Tmp Result
    X_tmp = similar(X)
    X_tmp .= X

    λ = get_threshold(opt)
    λ = issorted(λ) ? λ : sort(λ)
    λs = zeros(eltype(λ), size(Y,2))

    if progress
        progress =  init_progress(opt, X, A, Y, length(λ)*progress_outer, progress_offset)
    else
        progress = nothing
    end

    obj = zero(eltype(X))
    objtmp = zero(eltype(X))
    sparsity = zero(eltype(X))
    sparsitytmp = zero(eltype(X))

    @views for (i,λi) in enumerate(λ)
        init!(X_tmp, opt, A, Y)

        opt(X_tmp, A, Y, λi, maxiter = maxiter, abstol = abstol, f = f, g = g)
        
        # Increasing the threshold makes no sense
        all(iszero(X_tmp)) && break


        for j in 1:size(Y, 2)
            if fg(X_tmp[:, j], A, Y[:, j], λi) < fg(X[:, j], A, Y[:, j], λi)
                λs[j] =λi
                X[:, j] .= X_tmp[:, j]
            end
        end

        if !isnothing(progress)
            sparsity, obj = f(X, A, Y, λi)
            sparsitytmp, objtmp = f(X_tmp, A, Y, λi)

            ProgressMeter.next!(
            progress;
            showvalues = [
                (:Threshold, λi), (Symbol("Best Objective"), obj), (Symbol("Best Sparsity"), sparsity),
                (Symbol("Current Objective"), objtmp), (Symbol("Current Sparsity"), sparsitytmp)
            ]
            )
        end
    end

    for i in 1:size(Y, 2)
        @views clip_by_threshold!(X[:, i], λs[i])
    end

    return λs
end
