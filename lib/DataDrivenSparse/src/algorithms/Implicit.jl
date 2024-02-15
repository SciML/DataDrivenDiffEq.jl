"""
$(TYPEDEF)
Optimizer for finding a sparse implicit relationship via alternating the left-hand side of the problem and
solving the explicit problem, as introduced [here](https://royalsocietypublishing.org/doi/10.1098/rspa.2020.0279).

```math
\\argmin_{x} \\|x\\|_0 ~s.t.~Ax= 0
```

# Fields
$(FIELDS)

# Example
```julia
ImplicitOptimizer(STLSQ())
ImplicitOptimizer(0.1f0, ADMM)
```
"""
mutable struct ImplicitOptimizer{T <: AbstractSparseRegressionAlgorithm} <:
               AbstractSparseRegressionAlgorithm
    """Explicit Optimizer"""
    optimizer::T

    function ImplicitOptimizer(threshold = 1e-1, opt = STLSQ)
        optimizer = opt(threshold)
        return new{typeof(optimizer)}(optimizer)
    end

    function ImplicitOptimizer(opt::T) where {T <: AbstractSparseRegressionAlgorithm}
        return new{T}(opt)
    end
end

Base.summary(opt::ImplicitOptimizer) = "Implicit Optimizer using " * summary(opt.optimizer)

get_threshold(opt::ImplicitOptimizer) = get_threshold(opt.optimizer)

function (x::ImplicitOptimizer)(X, Y;
        options::DataDrivenCommonOptions = DataDrivenCommonOptions(),
        necessary_idx = ones(Bool, size(X, 1)),
        kwargs...)
    @unpack optimizer = x
    @unpack verbose = options

    n, _ = size(X)

    x_opt = zeros(eltype(X), 1, n) # All solutions
    inds = [false for _ in 1:n]

    solver = SparseLinearSolver(optimizer, options = options)

    results = Vector{Any}(undef, n)

    foreach(1:n) do i
        inds .= true
        inds[i] = false
        if verbose
            if i > 1
                @printf "\n"
            end
            @printf "Starting implicit sparse regression on possibility variable %6d of %6d\n" i n
        end
        results[i] = solver(X[inds, :], X[i, :])
    end

    # Find best results with dof >= 2
    best_id = 0
    foreach(enumerate(results)) do (i, res)
        inds .= true
        inds[i] = false
        if dof(first(res)) >= 2 && any(!iszero(coef(first(res))[:, necessary_idx[inds]]))
            if best_id <= 0
                best_id = i
            elseif aicc(first(res)) < aicc(first(results[best_id]))
                best_id = i
            end
        end
    end
    # Build the coefficient matrix
    inds .= true
    inds[best_id] = false
    best_cache, optimal_threshold, optimal_iterations = results[best_id]

    # Create the coefficient matrix
    x_opt[1, best_id] = -one(eltype(X))
    x_opt[1:1, inds] .= coef(best_cache)
    return x_opt, optimal_threshold, optimal_iterations
end
