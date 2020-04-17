abstract type AbstractKoopmanAlgorithm end;

mutable struct DMDPINV <: AbstractKoopmanAlgorithm end;

# Fast but more allocations
(x::DMDPINV)(X::AbstractArray, Y::AbstractArray) = Y / X

mutable struct DMDSVD{T} <: AbstractKoopmanAlgorithm
    rtol::T
end;

DMDSVD() = DMDSVD(0.0)

# Slower but less allocations
function (x::DMDSVD)(X::AbstractArray, Y::AbstractArray)
    U, S, V = svd(X)
    if zero(x.rtol) < x.rtol < one(x.rtol)
        r = vec(S .> x.rtol*maximum(S))
        U = U[:, r]
        S = S[r]
        V = V[:, r]
    end

    return Y*V*Diagonal(one(eltype(X)) ./ S)*U'
end
