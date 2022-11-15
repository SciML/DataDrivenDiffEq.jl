struct STLSQ{T <: Union{Number, AbstractVector}} <: AbstractSparseRegressionAlgorithm
    thresholds::T

    function STLSQ(threshold::T = 1e-1) where {T}
        @assert all(threshold .> zero(eltype(threshold))) "Threshold must be positive definite"

        return new{T}(threshold)
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

    λ = minimum(get_thresholds(alg))

    proximal = get_proximal(alg)

    if n_x <= m_x
        X = A * A'
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
