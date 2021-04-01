function truncated_svd(A::AbstractMatrix{T}, truncation::Real) where T <: Number
    truncation = min(truncation, one(T))
    U, S, V = svd(A)
    r = vec(S .> truncation*maximum(S))
    U = U[:, r]
    S = S[r]
    V = V[:, r]
    return U, S, V
end

function truncated_svd(A::AbstractMatrix, truncation::Int)
    U, S, V = svd(A)
    r = min(length(S), truncation)
    U = U[:, 1:r]
    S = S[1:r]
    V = V[:, 1:r]
    return U, S, V
end

"""
$(TYPEDEF)

Approximates the Koopman operator `K` based on

```julia
K = Y / X
```

where `Y` and `X` are data matrices. Returns a  `Eigen` factorization of the operator.

# Fields
$(FIELDS)

"""
mutable struct DMDPINV <: AbstractKoopmanAlgorithm end;

# Fast but more allocations
function (x::DMDPINV)(X::AbstractArray, Y::AbstractArray)
     K = Y / X
     return eigen(K)
 end


"""
$(TYPEDEF)

Approximates the Koopman operator `K` based on the singular value decomposition
of `X` such that:

```julia
K = Y*V*Σ*U'
```

where `Y` and `X = U*Σ*V'` are data matrices. The singular value decomposition is truncated via
the `truncation` parameter, which can either be an `Int` indiciating an index based truncation or a `Real`
indiciating a tolerance based truncation. Returns a `Eigen` factorization of the operator.

"""
mutable struct DMDSVD{T} <: AbstractKoopmanAlgorithm where T <: Number
    """Indiciates the truncation"""
    truncation::T
end;

DMDSVD() = DMDSVD(0.0)

# Slower but fewer allocations
function (x::DMDSVD{T})(X::AbstractArray, Y::AbstractArray) where T <: Real
    U, S, V = truncated_svd(X, x.truncation)
    xone = one(eltype(X))
    # Computed the reduced operator
    Sinv = Diagonal(xone ./ S)
    B = Y*V*Sinv
    Ã = U'B
    # Compute the modes
    λ, ω = eigen(Ã)
    φ = Diagonal(xone ./ λ)*B*ω
    return Eigen(λ, φ)
end

"""
    TOTALDMD(rtol, alg)

Approximates the Koopman operator `K` with the algorithm `alg` over the rank-reduced data
matrices `Xᵣ = X Qᵣ` and `Yᵣ = Y Qᵣ`, where `Qᵣ` originates from the singular value decomposition of
the joint data `Z = [X; Y]`. Based on [this paper](http://cwrowley.princeton.edu/papers/Hemati-2017a.pdf).

If `rtol` ∈ (0, 1) is given, the singular value decomposition is reduced to include only
entries bigger than `rtol*maximum(Σ)`. If `rtol` is an integer, the reduced SVD up to `rtol` is used
for computation.
"""
mutable struct TOTALDMD{R, A} <: AbstractKoopmanAlgorithm where {R <: Number, A <: AbstractKoopmanAlgorithm}
    truncation::R
    alg::A
end

TOTALDMD() = TOTALDMD(0.0, DMDPINV())

function (x::TOTALDMD)(X::AbstractArray, Y::AbstractArray)
    _ , _, Q = truncated_svd([X; Y], x.truncation)
    return x.alg(X*Q, Y*Q)
end
