struct STLSQ{T <: Union{Number, AbstractVector}} <: AbstractSparseRegressionAlgorithm
    thresholds::T

    function STLSQ(threshold::T = 1e-1) where {T}
        @assert all(threshold .> zero(eltype(threshold))) "Threshold must be positive definite"
        
        return new{T}(threshold)
    end
end

struct STLSQCache{usenormal, C <: AbstractArray, A <: BitArray, AT, BT} <: AbstractSparseRegressionCache
    X::C
    X_prev::C
    active_set::A
    proximal::SoftThreshold
    A::AT
    B::BT
end

init_cache(alg::STLSQ, A::AbstractMatrix, b::AbstractVector) = init_cache(alg, A, permutedims(b))

function init_cache(alg::STLSQ, A::AbstractMatrix, B::AbstractMatrix)
    n_x, m_x = size(A)
    @assert size(B, 1) == 1 "Caches only hold single targets!"

    λ = minimum(get_thresholds(alg))

    proximal = get_proximal(alg)

    if n_x <= m_x
        X = A*A'
        Y = B*A'
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

    return STLSQCache{usenormal, typeof(coefficients), typeof(active_set), typeof(X), typeof(Y)}(
        coefficients, prev_coefficients, 
        active_set, get_proximal(alg),
        X, Y
    )
end

function reset!(alg::STLSQ, cache::STLSQCache{_usenormal}, A::AbstractMatrix, B::AbstractVector) where {_usenormal}
    n_x, m_x = size(A)

    B = reshape(B, 1, length(B))

    # Fat
    if n_x <= m_x
        usenormal = true
        usenormal != _usenormal && return init_cache(alg, A, B)
        cache.A = A*A'
        cache.B = B*A'
    # Skinny
    else
        usenormal = false
        usenormal != _usenormal && return init_cache(alg, A, B)
        cache.A = A
        cache.B = B
    end
    
    λ = minimum(get_thresholds(alg))
    cache.X = /(A, B)
    cache.X_prev = zero(cache.X)
    
    active_set!(
        cache.active_set, cache.proximal, 
        cache.X, λ
    )

    return cache
end

function step!(cache::STLSQCache, λ::T) where T
    @unpack X, X_prev, active_set, proximal, A, B = cache

    X_prev .= X

    _regress!(cache)

    proximal(X, active_set, λ)
    return
end

function _regress!(cache::STLSQCache{true})
    @unpack X, A, B, active_set = cache
    p = vec(active_set)
    X[1:1,p] .= /(B[1:1, p], A[p, p])
    return 
end

function _regress!(cache::STLSQCache{false})
    @unpack X, A, B, active_set = cache
    p = vec(active_set)
    X[1:1, p] .= /(B, A[p, :])
    return 
end

