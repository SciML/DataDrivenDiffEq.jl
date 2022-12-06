"""
$(TYPEDEF)
`STLSQ` is taken from the [original paper on SINDY](https://www.pnas.org/content/113/15/3932) and implements a
sequentially thresholded least squares iteration. `λ` is the threshold of the iteration.
It is based upon [this matlab implementation](https://github.com/eurika-kaiser/SINDY-MPC/utils/sparsifyDynamics.m).
It solves the following problem
```math
\\argmin_{x} \\frac{1}{2} \\| Ax-b\\|_2 + \\rho \\|x\\|_2
```
with the additional constraint

```math
\\lvert x_i \\rvert > \\lambda
```

If the parameter `ρ > 0`, ridge regression will be performed using the normal equations of the corresponding 
regression problem.

# Fields
$(FIELDS)

# Example
```julia
opt = STLSQ()
opt = STLSQ(1e-1)
opt = STLSQ(1e-1, 1.0) # Set rho to 1.0
opt = STLSQ(Float32[1e-2; 1e-1])
```
## Note
This was formally `STRRidge` and has been renamed.
"""
struct STLSQ{T <: Union{Number, AbstractVector}, R <: Number} <:
       AbstractSparseRegressionAlgorithm
    """Sparsity threshold"""
    thresholds::T
    """Ridge regression parameter"""
    rho::R

    function STLSQ(threshold::T = 1e-1, rho::R = zero(eltype(T))) where {T, R <: Number}
        @assert all(threshold .> zero(eltype(threshold))) "Threshold must be positive definite"
        @assert rho>=zero(R) "Ridge regression parameter must be positive definite!"
        return new{T, R}(threshold, rho)
    end
end

Base.summary(::STLSQ) = "STLSQ"

struct STLSQCache{usenormal, C <: AbstractArray, A <: BitArray, AT, BT, ATT, BTT} <:
       AbstractSparseRegressionCache
    X::C
    X_prev::C
    active_set::A
    proximal::SoftThreshold
    A::AT
    B::BT
    # Original Data
    Ã::ATT
    B̃::BTT
end

function init_cache(alg::STLSQ, A::AbstractMatrix, b::AbstractVector)
    init_cache(alg, A, permutedims(b))
end

function init_cache(alg::STLSQ, A::AbstractMatrix, B::AbstractMatrix)
    n_x, m_x = size(A)
    @assert size(B, 1)==1 "Caches only hold single targets!"
    @unpack rho = alg
    λ = minimum(get_thresholds(alg))

    proximal = get_proximal(alg)

    if n_x <= m_x && !iszero(rho)
        X = A * A' + rho * I
        Y = B * A'
        usenormal = true
    else
        usenormal = false
        X = A
        Y = B
    end

    coefficients = Y / X

    prev_coefficients = zero(coefficients)

    active_set = BitArray(undef, size(coefficients))

    active_set!(active_set, proximal, coefficients, λ)

    return STLSQCache{usenormal, typeof(coefficients), typeof(active_set), typeof(X),
                      typeof(Y), typeof(A), typeof(B)}(coefficients, prev_coefficients,
                                                       active_set, get_proximal(alg),
                                                       X, Y, A, B)
end

function step!(cache::STLSQCache, λ::T) where {T}
    @unpack X, X_prev, active_set, proximal = cache

    X_prev .= X

    step!(cache)

    proximal(X, active_set, λ)
    return
end

function step!(cache::STLSQCache{true})
    @unpack X, A, B, active_set = cache
    p = vec(active_set)
    X[1:1, p] .= /(B[1:1, p], A[p, p])
    return
end

function step!(cache::STLSQCache{false})
    @unpack X, A, B, active_set = cache
    p = vec(active_set)
    X[1:1, p] .= /(B, A[p, :])
    return
end
