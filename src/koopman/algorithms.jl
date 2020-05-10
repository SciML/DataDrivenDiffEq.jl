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
entries bigger than `rtol*maximum(Σ)`. If `rtol` is an integer, the reduced SVD up to `rtol` is used
for computation.
"""
mutable struct DMDSVD{T} <: AbstractKoopmanAlgorithm
    rtol::T
end;

DMDSVD() = DMDSVD(0.0)

# Slower but less allocations
function (x::DMDSVD)(X::AbstractArray, Y::AbstractArray)
    U, S, V = svd(X)
    if typeof(x.rtol) <: Int && x.rtol < size(X, 1)
        V = V[:, x.rtol]
    elseif zero(x.rtol) < x.rtol < one(x.rtol)
        r = vec(S .> x.rtol*maximum(S))
        U = U[:, r]
        S = S[r]
        V = V[:, r]
    end

    return Y*V*Diagonal(one(eltype(X)) ./ S)*U'
end

"""
    TOTALDMD(rtol, alg)

Approximates the koopman operator `K` with the algorithm `alg` over the rank reduced data
matrices `Xᵣ = X Qᵣ` and `Yᵣ = Y Qᵣ` where `Qᵣ` originates from the singular value decomposition of
the joint data `Z = [X; Y]`. Based on [this paper](http://cwrowley.princeton.edu/papers/Hemati-2017a.pdf).

If `rtol` ∈ (0, 1) is given, the singular value decomposition is reduced to include only
entries bigger than `rtol*maximum(Σ)`. If `rtol` is an integer, the reduced SVD up to `rtol` is used
for computation.
"""
mutable struct TOTALDMD{R, A} <: AbstractKoopmanAlgorithm
    rtol::R
    alg::A
end

TOTALDMD() = TOTALDMD(0.0, DMDPINV())

function (x::TOTALDMD)(X::AbstractArray, Y::AbstractArray)
    _ , S, Q = svd([X; Y])
    if typeof(x.rtol) <: Int && x.rtol < size(X, 1)
        Q = Q[:, 1:x.rtol]
    elseif zero(x.rtol) < x.rtol < one(x.rtol)
        r = vec(S .> x.rtol*maximum(S))
        Q = Q[:, r]
    end

    return x.alg(X*Q, Y*Q)
end
