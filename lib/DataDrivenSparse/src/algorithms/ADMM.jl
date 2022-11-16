"""
$(TYPEDEF)
`ADMM` is an implementation of Lasso using the alternating direction methods of multipliers and
loosely based on [this implementation](https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html).
It solves the following problem
```math
\\argmin_{x} \\frac{1}{2} \\| Ax-b\\|_2 + \\lambda \\|x\\|_1
```
# Fields
$(FIELDS)
# Example
```julia
opt = ADMM()
opt = ADMM(1e-1, 2.0)
```
"""
mutable struct ADMM{T, R <: Number} <: AbstractSparseRegressionAlgorithm
    """Sparsity threshold parameter"""
    thresholds::T
    """Augmented Lagrangian parameter"""
    rho::R

    function ADMM(threshold::T = 1e-1, ρ::R = 1.0) where {T, R}
        @assert all(threshold .> zero(eltype(threshold))) "Threshold must be positive definite"
        @assert zero(R)<ρ "Augemented lagrangian parameter should be positive definite"
        return new{T, R}(threshold, ρ)
    end
end

Base.summary(::ADMM) = "ADMM"

struct ADMMCache{fat, C, A, AT, BT, T, ATT, BTT} <: AbstractSparseRegressionCache
    X::C
    X_prev::C
    active_set::A
    proximal::SoftThreshold
    #
    alpha::C
    w::C
    #
    A::AT
    B::BT
    rho::T
    # Original Data
    Ã::ATT
    B̃::BTT
end

function init_cache(alg::ADMM, A::AbstractMatrix, b::AbstractVector)
    init_cache(alg, A, permutedims(b))
end

function init_cache(alg::ADMM, A::AbstractMatrix, B::AbstractMatrix)
    n_x, m_x = size(A)

    @assert size(B, 1)==1 "Caches only hold single targets!"

    λ = minimum(get_thresholds(alg))

    @unpack rho = alg

    Y = B * A'
    if n_x <= m_x # This is skinny -> more measurements
        X = cholesky(A * A' .+ rho * I(size(A, 1)))
        fat = false
        coefficients = Y / X
    else # this is fat -> more basis elements
        X = cholesky(A' * A ./ rho .+ I(size(A, 2)))
        fat = true
        coefficients = B / A
    end

    proximal = SoftThreshold()

    idx = BitArray(undef, size(coefficients)...)

    active_set!(idx, proximal, coefficients, λ / rho)

    return ADMMCache{fat, typeof(coefficients), typeof(idx), typeof(X), typeof(Y),
                     typeof(rho), typeof(A), typeof(B)}(coefficients, zero(coefficients),
                                                        idx, proximal,
                                                        zero(coefficients),
                                                        zero(coefficients),
                                                        X, Y, rho, A, B)
end

# Fat regression
function step!(cache::ADMMCache{false}, λ::T) where {T <: Number}
    @unpack X, X_prev, active_set, proximal, A, B, w, alpha, rho = cache

    X_prev .= X

    X .= (B .+ rho .* (alpha .- w)) / A
    
    proximal(alpha, X .+ w, λ / rho)

    w .+= X .- alpha

    proximal(X, active_set, λ / rho)

    return
end

function step!(cache::ADMMCache{true}, λ::T) where {T <: Number}
    @unpack X, X_prev, active_set, proximal, A, B, Ã, w, alpha, rho = cache

    X_prev .= X

    q = (B .+ rho .* (alpha .- w))
    b = (((q * Ã) / A) * Ã' ./ rho^2)
    X .= q ./ rho .- b

    proximal(alpha, X .+ w, λ / rho)

    w .+= X .- alpha

    proximal(X, active_set, λ / rho)

    return
end
