# Tolerance
function truncated_svd(A::AbstractMatrix{T}, truncation::Real) where T <: Number
    truncation = min(truncation, one(T))
    U, S, V = svd(A)
    r = vec(S .> truncation*maximum(S))
    U = U[:, r]
    S = S[r]
    V = V[:, r]
    return U, S, V
end

# Explicit rank
function truncated_svd(A::AbstractMatrix{T}, truncation::Int) where T <: Number
    U, S, V = svd(A)
    r = [((i <= truncation && S[i] > zero(T)) ? true : false) for i in 1:length(S)]
    U = U[:, r]
    S = S[r]
    V = V[:, r]
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

# Signatures
$(SIGNATURES)
"""
mutable struct DMDPINV <: AbstractKoopmanAlgorithm end;

# Fast but more allocations
function (x::DMDPINV)(X::AbstractArray, Y::AbstractArray)
     K = Y / X
     return eigen(K)
 end

# DMDC
function (x::DMDPINV)(X::AbstractArray, Y::AbstractArray, U::AbstractArray)
    nx, m = size(X)
    nu, m = size(U)

    K̃ = Y / [X;U]

    K = K̃[:, 1:nx]
    B = K̃[:, nx+1:end]

    return eigen(K), B
end

# DMDC
function (x::DMDPINV)(X::AbstractArray, Y::AbstractArray, U::AbstractArray, B::AbstractArray)
    return x(X, Y-B*U), B
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
    φ = B*ω
    return Eigen(λ, φ)
end

# DMDc
function (x::DMDSVD{T})(X::AbstractArray, Y::AbstractArray, U::AbstractArray) where T <: Real
    nx, m = size(X)
    nu, m = size(U)
    # Input space svd
    Ũ, S̃, Ṽ = truncated_svd([X;U], x.truncation)
    # Output space svd
    Û, Ŝ, V̂ = svd(Y)

    # Split the svd
    U₁, U₂ = Ũ[1:nx,:], Ũ[nx+1:end,:]

    xone = one(eltype(X))
    # Computed the reduced operator
    C = Y*Ṽ*Diagonal(xone ./ S̃ ) # Common submatrix
    # We do not project onto a reduced subspace here.
    # This would mess up our initial conditions, since sometimes we have
    # x1->x2, x2->x1
    Ã = Û'C*U₁'Û
    B̃ = C*U₂'
    # Compute the modes
    λ, ω = eigen(Ã)
    φ = C*U₁'Û*ω
    return Eigen(λ, φ), B̃
end

function (x::DMDSVD{T})(X::AbstractArray, Y::AbstractArray, U::AbstractArray, B::AbstractArray) where T <: Real
    return x(X, Y-B*U), B
end


"""
$(TYPEDEF)

Approximates the Koopman operator `K` based on the singular value decomposition
of `X` and `Y` such that:

```julia
K₁ = Y*V₁*Σ₁*U₁'
K₂ = X*V₂*Σ₂*U₂'
K = (K₁*inv(K₂))^(0.5)
```

where `Y = U₂*Σ₂*V₂'` and `X = U₁*Σ₁*V₁'` are data matrices. The singular value decomposition is truncated via
the `truncation` parameter, which can either be an `Int` indiciating an index based truncation or a `Real`
indiciating a tolerance based truncation. Returns a `Eigen` factorization of the operator.


# Fields
$(FIELDS)

# Signatures
$(SIGNATURES)
"""
mutable struct fbDMD{T} <: AbstractKoopmanAlgorithm where T <: Number
    """Indiciates the truncation"""
    truncation::T
end;
fbDMD() = fbDMD(0.0)
function (x::fbDMD{T})(X::AbstractArray, Y::AbstractArray) where T <: Real
    U₁, S₁, V₁ = truncated_svd(X, x.truncation)
    xone = one(eltype(X))
    # Computed the reduced operator
    Sinv₁ = Diagonal(xone ./ S₁)
    B₁ = Y*V₁*Sinv₁
    A₁ = U₁'B₁  # forward DMD

    U₂, S₂, V₂ = truncated_svd(Y, x.truncation)
    yone = one(eltype(Y))
    # Computed the reduced operator
    Sinv₂ = Diagonal(yone ./ S₂)
    B₂ = X*V₂*Sinv₂
    A₂ = U₂'B₂  # Backward DMD

    Ã = (A₁/A₂)^(0.5)
    # Compute the modes
    λ, ω = eigen(Ã)
    φ = B₁*ω
    return Eigen(λ, φ)
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
mutable struct TOTALDMD{R, A} <: AbstractKoopmanAlgorithm where {R <: Number, A <: AbstractKoopmanAlgorithm}
    truncation::R
    alg::A
end

TOTALDMD() = TOTALDMD(0.0, DMDPINV())

function (x::TOTALDMD)(X::AbstractArray, Y::AbstractArray)
    _ , _, Q = truncated_svd([X; Y], x.truncation)
    return x.alg(X*Q, Y*Q)
end

function (x::TOTALDMD)(X::AbstractArray, Y::AbstractArray, U::AbstractArray)
    _ , _, Q = truncated_svd([X; Y], x.truncation)
    return x.alg(X*Q, Y*Q, U*Q)
end


function (x::TOTALDMD)(X::AbstractArray, Y::AbstractArray, U::AbstractArray, B::AbstractArray)
    _ , _, Q = truncated_svd([X; Y], x.truncation)
    return x.alg(X*Q, (Y-B*U)*Q), B
end
