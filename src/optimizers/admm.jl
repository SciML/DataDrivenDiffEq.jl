"""
$(TYPEDEF)
`ADMM` is an implementation of Lasso using the alternating direction methods of multipliers and
loosely based on [this implementation](https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html).
It solves the following problem
```math
\\min_{x} \\frac{1}{2} \\| Ax-b\\|_2 + \\lambda \\|x\\|_1
```
#Fields
$(FIELDS)
# Example
```julia
opt = ADMM()
opt = ADMM(1e-1, 2.0)
```
"""
mutable struct ADMM{T, R} <: AbstractOptimizer{T}
    """Sparsity threshold"""
    λ::T
    """Augmented Lagrangian parameter"""
    ρ::R


    function ADMM(threshold::T = 1e-1, ρ::R = 1.0) where {T, R}
        @assert all(threshold .> zero(eltype(threshold))) "Threshold must be positive definite"
        @assert zero(R) < ρ "Augemented lagrangian parameter should be positive definite"
        return new{T, R}(threshold, ρ)
    end
end

function (opt::ADMM{T,H})(X, A, Y, λ::U = first(opt.λ);
    maxiter::Int64 = maximum(size(A)), abstol::U = eps(eltype(T)), progress = nothing)  where {T, H, U}

    n, m = size(A)

    ρ = opt.ρ

    x_i = zero(X)
    u = zero(X)
    z = zero(X)

    @views x_i .= X

    P = A'A .+ Diagonal(ρ .* ones(eltype(X),m))
    P = cholesky!(P)
    c = A'*Y

    R = SoftThreshold()

    iters = 0
    converged = false

    xzero = zero(eltype(X))
    obj = xzero
    sparsity = xzero
    conv_measure = xzero

    _progress = isa(progress, Progress)
    initial_prog = _progress ? progress.counter : 0



    while (iters < maxiter) && !converged
        iters += 1

        @views ldiv!(z, P, c .+ ρ .* (z .- u))
        @views R(X, z .+ u, λ ./ ρ)
        @views u .= u .+ z .- X

        @views conv_measure = norm(x_i .- X, 2)

        if _progress
            @views obj = norm(Y .- A*X, 2)
            @views sparsity = norm(X, 0)

            ProgressMeter.next!(
            progress;
            showvalues = [
                (:Threshold, λ), (:Objective, obj), (:Sparsity, sparsity),
                (:Convergence, conv_measure)
            ]
            )
        end


        if conv_measure < abstol
            converged = true

            if _progress

                ProgressMeter.update!(
                progress,
                initial_prog + maxiter
                )
            end

        else
            @views x_i .= X
        end
    end
    @views clip_by_threshold!(X, λ)
    return
end
