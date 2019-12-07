
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
    return Koopman(Ã,Λ,ω,W, nothing, nothing, :CompanionDMD)
end
