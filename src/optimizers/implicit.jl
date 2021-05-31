"""
$(TYPEDEF)
Optimizer for finding a sparse implicit relationship via alternating the left hand side of the problem and
solving the explicit problem, as introduced [here]().

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
mutable struct ImplicitOptimizer{T} <: AbstractSubspaceOptimizer{T}
    """Explicit Optimizer"""
    o::AbstractOptimizer{T}

    function ImplicitOptimizer(threshold = 1e-1, opt = STLSQ)
        return new{typeof(threshold)}(opt(threshold))
    end

    function ImplicitOptimizer(opt::AbstractOptimizer{T}) where T
        return new{T}(opt)
    end

end


Base.summary(opt::ImplicitOptimizer) = "Implicit Optimizer using "*summary(opt.o)

get_threshold(opt::ImplicitOptimizer) = get_threshold(opt.o)

function (opt::ImplicitOptimizer{T})(X, A, Y, λ::V = first(opt.o.λ);
    maxiter::Int64 = maximum(size(A)), abstol::V = eps(eltype(T)),
    rtol::V = zero(eltype(T)) ,progress = nothing,
    f::Function = F(opt),
    g::Function = G(opt))  where {T, V}

    exopt = opt.o

    n,m = size(A)
    ny, my = size(Y)
    nx, mx = size(X)
    nq, mq = 0,0

    # Closure for the pareto function
    fg(x, A, y) = (g∘f)(x, A, y)
    fg(x, A) = (g∘f)(x,A)

    xzero = zero(eltype(X))
    xone = one(eltype(X))

    obj = xzero
    sparsity = xzero
    conv_measure = xzero

    iters = 0
    converged = false

    # Build a quadratic matrix
    x_tmp = zeros(eltype(X), m, m) # The temporary solution
    x_opt = zeros(eltype(X), m, m) # All solutions
    inds = [false for _ in 1:m]

    _progress = isa(progress, Progress)

    # Set progress -1 and
    initial_prog = _progress ? progress.counter : 0

    # Iterate over all columns of A ( which represent the lhs )
    @views for j in 1:m
        inds .= true
        inds[j] = false
        x_tmp[j, j] = -one(eltype(X)) # We set this to -1 for the lhs
        # Solve explicit problem
        x_tmp[inds, j:j] .= init(exopt, A[:, inds], A[:, j:j])

        # Use optimizer
        λs = @views sparse_regression!(x_tmp[inds, j:j], A[:, inds], A[:, j:j],exopt,
            f = f, g = g, maxiter = maxiter, abstol = abstol)

        # Normalize
        x_tmp[j,j] = maximum(x_tmp[inds, j:j])
        x_tmp[inds, j:j] ./= x_tmp[j,j]
        #x_tmp[:,j:j] .= x_tmp[:, j:j] ./ maximum(x_tmp[:, j:j])
        if _progress
            sparsity, obj = f(x_tmp[inds, :], A[:, inds], A[:, j:j], λs[1])

            ProgressMeter.next!(
            progress;
            showvalues = [
                (:Threshold, λ), (:Objective, obj), (:Sparsity, sparsity)
                ]
                )
        end
    end


    # Reduce the solution size to linear independent columns
    @views x_tmp = linear_independent_columns(x_tmp, rtol)

    # Indicate if already used
    _included = zeros(Bool, my, size(x_tmp, 2))
    for i in 1:my, j in 1:size(x_tmp, 2)
        # Check, if already included
        any(_included[:, j]) && continue
        # Selector
        inds .= true; inds[j] = false
        
        if fg(x_tmp[inds, j], A[:, inds], A[:, j]) < fg(X[inds, i], A[:, inds], A[:, j])
            X[:, i] .= x_tmp[:, i]
            _included[i,j] = true
        end
        
    end

    if rank(X'X) < my
        @warn "$opt @ $λ has found illconditioned equations. Vary the threshold or relative tolerance."
    end

    return
end
