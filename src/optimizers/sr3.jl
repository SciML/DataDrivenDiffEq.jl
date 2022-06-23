#Based upon alg 2 in https://ieeexplore.ieee.org/document/8573778

"""
$(TYPEDEF)
`SR3` is an optimizer framework introduced [by Zheng et. al., 2018](https://ieeexplore.ieee.org/document/8573778) and used within
[Champion et. al., 2019](https://arxiv.org/abs/1906.10612). `SR3` contains a sparsification parameter `λ`, a relaxation `ν`.
It solves the following problem
```math
\\argmin_{x, w} \\frac{1}{2} \\| Ax-b\\|_2 + \\lambda R(w) + \\frac{\\nu}{2}\\|x-w\\|_2
```
Where `R` is a proximal operator and the result is given by `w`.

# Fields
$(FIELDS)

# Example
```julia
opt = SR3()
opt = SR3(1e-2)
opt = SR3(1e-3, 1.0)
opt = SR3(1e-3, 1.0, SoftThreshold())
```
## Note
Opposed to the original formulation, we use `ν` as a relaxation parameter,
as given in [Champion et. al., 2019](https://arxiv.org/abs/1906.10612). In the standard case of
hard thresholding the sparsity is interpreted as `λ = threshold^2 / 2`, otherwise `λ = threshold`.
"""
mutable struct SR3{T,V,P<:AbstractProximalOperator} <: AbstractOptimizer{T}
    """Sparsity threshold"""
    λ::T
    """Relaxation parameter"""
    ν::V
    """Proximal operator"""
    R::P

    function SR3(
        threshold::T = 1e-1,
        ν::V = 1.0,
        R::P = HardThreshold(),
    ) where {T,V<:Number,P<:AbstractProximalOperator}
        @assert all(threshold .> zero(eltype(threshold))) "Threshold must be positive definite"
        @assert ν > zero(V) "Relaxation must be positive definite"

        λ = isa(R, HardThreshold) ? threshold .^ 2 / 2 : threshold
        return new{typeof(λ),V,P}(λ, ν, R)
    end

    function SR3(threshold::T, R::P) where {T,P<:AbstractProximalOperator}
        @assert all(threshold .> zero(eltype(threshold))) "Threshold must be positive definite"
        λ = isa(R, HardThreshold) ? threshold .^ 2 / 2 : threshold
        ν = one(eltype(λ))
        return new{typeof(λ),eltype(λ),P}(λ, ν, R)
    end
end

Base.summary(::SR3) = "SR3"

function (opt::SR3{T,V,R})(
    X,
    A,
    Y,
    λ::V = first(opt.λ);
    maxiter::Int64 = maximum(size(A)),
    abstol::V = eps(eltype(T)),
    progress = nothing,
    kwargs...,
) where {T,V,R}

    n, m = size(A)
    ν = opt.ν
    W = copy(X)

    # Init matrices
    H = A' * A + I(m) * ν
    H = cholesky!(H)
    X̂ = A' * Y

    w_i = similar(W)
    w_i .= W
    iters = 0

    iters = 0
    converged = false

    xzero = zero(eltype(X))
    obj = xzero
    sparsity = xzero
    conv_measure = xzero

    _progress = isa(progress, Progress)

    @views while (iters < maxiter) && !converged
        iters += 1

        # Solve ridge regression
        X .= H \ (X̂ .+ W * ν)
        # Proximal
        opt.R(W, X, λ)

        conv_measure = norm(w_i .- W, 2)

        if _progress
            obj = norm(Y - A * X, 2)
            sparsity = norm(X, 0, λ)

            ProgressMeter.next!(
                progress;
                showvalues = [
                    (:Threshold, λ),
                    (:Objective, obj),
                    (:Sparsity, sparsity),
                    (:Convergence, conv_measure),
                ],
            )
        end


        if conv_measure < abstol
            converged = true
        else
            w_i .= W
        end
    end
    # We really search for W here
    X .= W
    @views clip_by_threshold!(X, λ)
    return
end
