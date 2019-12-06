using LinearAlgebra
using DataDrivenDiffEq

function CompanionMatrixDMD(X::AbstractArray; dt::T = 0.0) where T <: Real
    return CompanionMatrixDMD(X[:, 1:end-1], X[:, end], dt = dt)
end

function CompanionMatrixDMD(X::AbstractArray, Y::AbstractArray; dt::T = 0.0)  where T <: Real
    @assert size(Y) == (size(X)[1], )
    @assert dt >= 0

    # State dimension
    nₓ = size(X)[1]
    # Build companion matrix
    c = pinv(X)*Y
    C = hcat([zeros(eltype(c), 1, length(c)-1); Diagonal(ones(eltype(c), length(c)-1))], c)

    # Get the eigendecomposition
    # Eigen Decomposition for solution
    Λ, W̃ = eigen(Matrix(C))

    # Convert to real modes
    W = X*W̃
    # Sort to get meaningfull modes
    ids = sortperm(norm.(eachcol(W), 2), rev = true)

    W = W[:, ids[1:nₓ]]
    Λ = Λ[ids[1:nₓ]]
    Ã = real.(W*Diagonal(Λ)*inv(W))

    if dt > 0.0
        # Casting Complex enforces results
        ω = log.(Complex.(Λ)) ./ Δt
    else
        ω = []
    end
    return ExactDMD(Ã,Λ,ω,W, nothing, nothing)
end

function hankel(a::AbstractArray, b::AbstractArray; scale::T = 1.0) where T <: Number
    p = vcat([a; b[2:end]])
    n = length(b)-1
    m = length(p)-n+1
    H = Array{eltype(p)}(undef, n, m)
    @inbounds for i in 1:n
        @inbounds for j in 1:m
            H[i,j] = p[i+j-1]
        end
    end
    mul!(H, norm(H[:, 1])/scale, H)
    return H
end

function HankelDMD(X::AbstractArray,  n::Int64, m::Int64)
    return HankelDMD(X[:, 1:end-1], X[:, 2:end], n, m)
end

function HankelDMD(X::AbstractArray, Y::AbstractArray, n::Int64 , m::Int64)
    @assert size(X)[2] >= n+m
    @assert size(X) == size(Y)
    # TODO Assert size a priori
    # Double check the method with examples
    H₀ = [] # Hankel matrices
    H₁ = []
    scale = norm(X[:, 1])
    @inbounds for i in 1:size(X)[1]
        push!(H₀, hankel(@view(X[i, 1:n]), @view(X[i, n:n+m]), scale = i > 1 ? scale : one(eltype(X))))
        push!(H₁, hankel(@view(Y[i, 1:n]), @view(Y[i, n:n+m]), scale = i > 1 ? scale : one(eltype(X))))
    end

    H₀ = hcat(H₀...)
    H₁ = hcat(H₁...)
    return ExactDMD(H₀, H₁)

end
