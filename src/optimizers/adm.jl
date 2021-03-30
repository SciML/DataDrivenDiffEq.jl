"""
$(TYPEDEF)
Optimizer for finding a sparse basis vector in a subspace based on [this paper](https://arxiv.org/pdf/1412.4659.pdf).
It solves the following problem
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

## Note
While useable for implicit problems, a better choice in general is given by the
`ImplicitOptimizer` which tends to be more robust. 
"""
mutable struct ADM{T} <: AbstractSubspaceOptimizer{T}
    """Sparsity threshold"""
    λ::T

    function ADM(threshold = 1e-1)
        @assert all(threshold .> zero(eltype(threshold))) "Threshold must be positive definite"

        return new{typeof(threshold)}(threshold)
    end
end

Base.summary(::ADM) = "ADM"

function (opt::ADM{T})(X, A, Y, λ::V = first(opt.λ);
    maxiter::Int64 = maximum(size(A)), abstol::V = eps(eltype(T)), progress = nothing,
    rtol::V = convert(eltype(T), 0.0),
    atol::V = convert(eltype(T), 0.99),
    f::Function = F(opt), g::Function = G(opt))  where {T, V}

    n,m = size(A)
    ny, my = size(Y)
    nx, mx = size(X)
    nq, mq = 0,0

    # Closure for the pareto function
    fg(x, A, y) = (g∘f)(x, A, y)
    fg(x, A) = (g∘f)(x,A)

    # Init all variables
    R = SoftThreshold()

    xzero = zero(eltype(X))
    obj = xzero
    sparsity = xzero
    conv_measure = xzero

    iters = 0
    converged = false

    max_ind = 0

    _progress = isa(progress, Progress)
    initial_prog = 0
    N = nullspace(A[:,:], atol = atol)
    Q = deepcopy(N)
    map(x->normalize!(x), eachrow(Q))
    Q_ = deepcopy(N)
    x = N'Q

    @views while (iters < maxiter) && !converged
        iters += 1



        R(x, N'Q, λ)
        mul!(Q, N, x)
        map(x->normalize!(x), eachrow(Q))

        conv_measure = norm(Q .- Q_, 2)

        if _progress
            sparsity, obj = f(Q,A,λ)

            ProgressMeter.next!(
            progress;
            showvalues = [
                (:Threshold, λ), (:Objective, obj), (:Sparsity, sparsity),
                (:Convergence, conv_measure),
                ]
                )
        end


        if conv_measure < abstol
            converged = true
        else
            Q_ .= Q
        end
    end

    clip_by_threshold!(Q, λ)

    # Reduce the solution size to linear independent columns
    qrx = qr(Q, Val(true))
    _r = abs.(diag(qrx.R))
    r = findlast(_r .>= rtol*first(_r))
    r = min(r, rank(Q))
    idx = sort(qrx.p[1:r])
    Q = Q[:, idx]

    # Indicate if already used
    _included = zeros(Bool, my, r)

    @views for i in 1:my, j in 1:r
        # Check, if already included
        any(_included[:, j]) && continue
        if evaluate_pareto!(X[:, i], Q[:,r] , fg, A, λ)
            _included[i,j] = true
        end
    end

    if _progress
        sparsity, obj = f(X,A,λ)


        ProgressMeter.update!(
            progress,
            initial_prog + maxiter -1
        )

        ProgressMeter.next!(
        progress;
        showvalues = [
            (:Threshold, λ), (:Objective, obj), (:Sparsity, sparsity),
            ]
        )
    end

    return
end
