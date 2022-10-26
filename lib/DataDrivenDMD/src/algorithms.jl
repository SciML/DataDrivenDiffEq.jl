function truncated_svd(A::AbstractMatrix{T}, truncation::Real) where {T <: Number}
    truncation = min(truncation, abs(one(T)))
    U, S, V = svd(A)
    r = vec(S .> truncation * maximum(S))
    U = U[:, r]
    S = S[r]
    V = V[:, r]
    return U, S, V
end

# Explicit rank
function truncated_svd(A::AbstractMatrix{T}, truncation::Int) where {T <: Number}
    U, S, V = svd(A)
    r = [((i <= truncation && S[i] > zero(T)) ? true : false) for i in eachindex(S)]
    U = U[:, r]
    S = S[r]
    V = V[:, r]
    return U, S, V
end

function (alg::AbstractKoopmanAlgorithm)(X::AbstractMatrix, Y::AbstractMatrix,
                                         U::AbstractMatrix, ::Nothing)
    n_x = size(X, 1)
    if !isempty(U)
        Z = hcat(X, U)
        K̃ = alg(Z, Y)
        K = K̃[:, 1:n_x]
        B = K̃[:, (n_x + 1):end]
    else
        K = alg(X, Y)
        B = Array{eltype(X)}(undef, 0, 0)
    end
    return K, B
end

function (alg::AbstractKoopmanAlgorithm)(X::AbstractMatrix, Y::AbstractMatrix,
                                         U::AbstractMatrix, B::AbstractMatrix)
    if !isempty(U) && !isempty(B)
        Z = X - B * U
        K = alg(Z, Y)
    else
        K = alg(X, Y)
    end
    return K, B
end

"""
$(TYPEDEF)

Approximates the [`Koopman`](@ref) by solving the linear system

```julia
Y = K X
```
via the backslash.

`Y` and `X` are data matrices. Returns an [`KoopmanResult`](@ref).
"""
struct DMDPINV <: AbstractKoopmanAlgorithm end

function (::DMDPINV)(X::AbstractArray, Y::AbstractArray)
    return Y / X
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
# Fields
$(FIELDS)
# Signatures
$(SIGNATURES)
"""
mutable struct DMDSVD{T} <: AbstractKoopmanAlgorithm where {T <: Number}
    """Indiciates the truncation"""
    truncation::T
end;

DMDSVD() = DMDSVD(0.0)

# Slower but fewer allocations
function (x::DMDSVD{T})(X::AbstractArray, Y::AbstractArray) where {T <: Real}
    U, S, V = truncated_svd(X, x.truncation)
    xone = one(eltype(X))
    # Computed the reduced operator
    Sinv = Diagonal(xone ./ S)
    B = Y * V * Sinv
    Ã = U'B
    # Compute the modes
    λ, ω = eigen(Ã)
    φ = B * ω
    K = Matrix(Eigen(λ, φ))
    eltype(X) <: Real && return real.(K)
    return K
end

function (x::DMDSVD{T})(X::AbstractArray, Y::AbstractArray,
                        U::AbstractArray) where {T <: Real}
    nx, m = size(X)
    nu, m = size(U)
    # Input space svd
    Ũ, S̃, Ṽ = truncated_svd([X; U], x.truncation)
    # Output space svd
    Û, _ = svd(Y)

    # Split the svd
    U₁, U₂ = Ũ[1:nx, :], Ũ[(nx + 1):end, :]

    xone = one(eltype(X))
    # Computed the reduced operator
    C = Y * Ṽ * Diagonal(xone ./ S̃) # Common submatrix
    # We do not project onto a reduced subspace here.
    # This would mess up our initial conditions, since sometimes we have
    # x1->x2, x2->x1
    Ã = Û'C * U₁'Û
    B̃ = C * U₂'
    # Compute the modes
    λ, ω = eigen(Ã)
    φ = C * U₁'Û * ω
    K = Matrix(Eigen(λ, φ))
    K = eltype(X) <: Real ? real.(K) : K

    return K, B̃
end

"""
$(TYPEDEF)
Approximates the Koopman operator `K` with the algorithm `alg` over the rank-reduced data
matrices `Xᵣ = X Qᵣ` and `Yᵣ = Y Qᵣ`, where `Qᵣ` originates from the singular value decomposition of
the joint data `Z = [X; Y]`. Based on [this paper](http://cwrowley.princeton.edu/papers/Hemati-2017a.pdf).
If `rtol` ∈ (0, 1) is given, the singular value decomposition is reduced to include only
entries bigger than `rtol*maximum(Σ)`. If `rtol` is an integer, the reduced SVD up to `rtol` is used
for computation.

# Fields
$(FIELDS)

# Signatures
$(SIGNATURES)
"""
mutable struct TOTALDMD{R, A} <:
               AbstractKoopmanAlgorithm where {R <: Number, A <: AbstractKoopmanAlgorithm}
    truncation::R
    alg::A
end

TOTALDMD() = TOTALDMD(0.0, DMDPINV())

function (x::TOTALDMD)(X::AbstractArray, Y::AbstractArray)
    _, _, Q = truncated_svd([X; Y], x.truncation)
    return x.alg(X * Q, Y * Q)
end

function (x::TOTALDMD)(X::AbstractArray, Y::AbstractArray, U::AbstractArray)
    _, _, Q = truncated_svd([X; Y], x.truncation)
    return x.alg(X * Q, Y * Q, U * Q)
end

function (x::TOTALDMD)(X::AbstractArray, Y::AbstractArray, U::AbstractArray,
                       B::AbstractArray)
    _, _, Q = truncated_svd([X; Y], x.truncation)
    K, _ = x.alg(X * Q, (Y - B * U) * Q)
    return (K, B)
end
