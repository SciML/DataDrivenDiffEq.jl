"""
$(TYPEDEF)
Optimizer for finding a sparse implicit relationship via alternating the left hand side of the problem and
solving the explicit problem, as introduced [here]().

```math
\\min_{x} \\|x\\|_0 ~s.t.~Ax= 0
```


#Fields
$(FIELDS)
# Example
```julia
ADM()
ADM(λ = 0.1)
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

    # ADM is already implicit
    function ImplicitOptimizer(opt::ADM)
        @info "ADM is already implict. Return ADM."
        return ADM
    end
end

get_threshold(opt::ImplicitOptimizer) = get_threshold(opt.o)

function (opt::ImplicitOptimizer{T})(X, A, Y, λ::V = first(opt.o.λ);
    maxiter::Int64 = maximum(size(A)), abstol::V = eps(eltype(T)), progress = nothing,
    f::Function = F(opt),
    g::Function = G(opt))  where {T, V}

    exopt = opt.o

    n,m = size(A)
    ny, my = size(Y)
    nx, mx = size(X)
    half_size = Int64(round(nx/2))
    nq, mq = 0,0

    # Closure for the pareto function
    fg(x, A) = (g∘f)(x, A)

    xzero = zero(eltype(X))
    xone = one(eltype(X))
    obj = xzero
    sparsity = xzero
    conv_measure = xzero

    iters = 0
    converged = false

    max_ind = 0

    nspaces = _assemble_ns(A, Y)
    nθ, mθ = size(nspaces[1])

    # Build a quadratic matrix
    x_tmp = zeros(eltype(X), mθ, 1)
    x_opt = zeros(eltype(X), mθ, mθ)
    inds = [false for _ in 1:mθ]

    for i in 1:size(nspaces, 1)
        # Current nullspace
        θ = nspaces[i]'

        # Set the current result to zero
        for j in 1:mθ
            inds .= true
            inds[j] = false
            x_tmp[j] = 1
            # Solve explicit problem
            @views x_tmp[inds, :] .= init(exopt, θ[inds, :]', θ[j:j, :]')
            @views exopt(x_tmp[inds, :], θ[inds, :]', θ[j:j, :]',λ, maxiter = maxiter, abstol = abstol)
            if j == 1
                X[:, i] .= x_tmp[:, 1]
            else
                evaluate_pareto!(X[:, i], x_tmp[:, 1] , fg, θ')
            end
        end
    end
    clip_by_threshold!(X, λ)
    return
end
