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

function hankel(A, m)
        if m == 0
                return A
        end

        x,y = size(A)

        H = Array{eltype(A)}(undef, x*(m+1), y-m)

        for i in 1:m+1
                H[1+(i-1)*x:i*x, :] = A[:,i:y-(m-i)-1]
        end
        return H
end

function HankelDMD(X, Y, m)
        # Delay embedding
        #D = delay_embedding(X, m)
        # Companion matrix
        c = pinv(X)*X[:, end]
        C = hcat([zeros(eltype(c), 1, length(c)-1); Diagonal(ones(eltype(c), length(c)-1))], c)
end

using Plots
A = rand(3,100)

c = HankelDMD(A[:, 1:end-1], A[:, 2:end], 1)
scatter(eigvals(Matrix(c)))
H = taken(A[:, 1:end-1], 1)
H2 = taken(A[:, 2:end],1)

H*pinv(H2)
