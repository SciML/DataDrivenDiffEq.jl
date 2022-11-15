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
Opposed to the original formulation, we use `nu` as a relaxation parameter,
as given in [Champion et. al., 2019](https://arxiv.org/abs/1906.10612). In the standard case of
hard thresholding the sparsity is interpreted as `λ = threshold^2 / 2`, otherwise `λ = threshold`.
"""
mutable struct SR3{T, V, P <: AbstractProximalOperator} <: AbstractSparseRegressionAlgorithm
    """Sparsity threshold"""
    thresholds::T
    """Relaxation parameter"""
    nu::V
    """Proximal operator"""
    proximal::P

    function SR3(threshold::T = 1e-1, nu::V = 1.0,
                 R::P = HardThreshold()) where {T, V <: Number,
                                                P <: AbstractProximalOperator}
        @assert all(threshold .> zero(eltype(threshold))) "Threshold must be positive definite"
        @assert nu>zero(V) "Relaxation must be positive definite"

        λ = isa(R, HardThreshold) ? threshold .^ 2 / 2 : threshold
        return new{typeof(λ), V, P}(λ, nu, R)
    end

    function SR3(threshold::T, R::P) where {T, P <: AbstractProximalOperator}
        @assert all(threshold .> zero(eltype(threshold))) "Threshold must be positive definite"
        λ = isa(R, HardThreshold) ? threshold .^ 2 / 2 : threshold
        ν = one(eltype(λ))
        return new{typeof(λ), eltype(λ), P}(λ, ν, R)
    end
end

Base.summary(::SR3) = "SR3"

struct SR3Cache{C, A, P <: AbstractProximalOperator, AT, BT, T, ATT, BTT} <:
       AbstractSparseRegressionCache
    X::C
    X_prev::C
    active_set::A
    proximal::P
    #
    W::C
    #
    A::AT
    B::BT
    nu::T
    # Original Data
    Ã::ATT
    B̃::BTT
end

function init_cache(alg::SR3, A::AbstractMatrix, b::AbstractVector)
    init_cache(alg, A, permutedims(b))
end

function init_cache(alg::SR3, A::AbstractMatrix, B::AbstractMatrix)
    n_x, m_x = size(A)

    @assert size(B, 1)==1 "Caches only hold single targets!"

    λ = minimum(get_thresholds(alg))

    @unpack nu, proximal = alg

    # Init matrices
    X = cholesky(A * A' .+ I(n_x) * nu)
    Y = B * A'

    coefficients = Y / X

    idx = BitArray(undef, size(coefficients)...)

    active_set!(idx, proximal, coefficients, λ)

    return SR3Cache{typeof(coefficients), typeof(idx), typeof(proximal), typeof(X),
                    typeof(Y), typeof(nu), typeof(A), typeof(B)}(coefficients,
                                                                 copy(coefficients), idx,
                                                                 proximal,
                                                                 zero(coefficients),
                                                                 X, Y, nu, A, B)
end

function step!(cache::SR3Cache, λ::T) where {T <: Number}
    @unpack X, X_prev, active_set, proximal, A, B, W, nu = cache

    X_prev .= X

    W .= (B .+ X * nu) / A

    proximal(X, W, λ)

    active_set!(active_set, proximal, X, λ)

    return
end
