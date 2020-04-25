abstract type AbstractKoopmanAlgorithm end;

"""
    DMDPINV()

Approximates the koopman operator `K` based on

```julia
K = Y / X
```

where `Y` and `X` are data matrices.

"""
mutable struct DMDPINV <: AbstractKoopmanAlgorithm end;

# Fast but more allocations
(x::DMDPINV)(X::AbstractArray, Y::AbstractArray) = Y / X


"""
    DMDSVD(rtol)

Approximates the koopman operator `K` based on the singular value decomposition
of `X` such that

```julia
K = Y*V*Σ*U'
```

where `Y` and `X = U*Σ*V'` are data matrices.
If `rtol` ∈ (0, 1) is given, the singular value decomposition is reduced to include only
entries bigger than `rtol*maximum(Σ)`.
"""
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
