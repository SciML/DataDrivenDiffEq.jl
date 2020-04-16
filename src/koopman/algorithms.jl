abstract type AbstractKoopmanAlgorithm end;

mutable struct DMDPINV <: AbstractKoopmanAlgorithm end;

# Fast but more allocations
(x::DMDPINV)(X::AbstractArray, Y::AbstractArray) = Y / X

mutable struct DMDSVD <: AbstractKoopmanAlgorithm end;

# Slower but less allocations
function (x::DMDSVD)(X::AbstractArray, Y::AbstractArray)
    U, S, V = svd(X)
    return Y*V*Diagonal(one(eltype(X)) ./ S)*U'
end
