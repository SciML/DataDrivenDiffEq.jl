mutable struct Koopman{M,L,W,F, Q, P} <: abstractKoopmanOperator
    Ã::M # Approximation of the operator
    λ::L # Eigenvalues (discrete time)
    ω::W # Frequencies
    ϕ::F # Modes


    # For online update
    Qₖ::Q
    Pₖ::P

    # Algorithm and method
    alg::Symbol
    method::Symbol
end

function koopman_pinv(X::AbstractArray, Y::AbstractArray, alg::Symbol, dt::T = 0.0) where T <: Real
    @assert dt >= 0 "Time step has to be positive semidefinite!"
    @assert size(Y)[1] .<= size(Y)[2]
    @assert size(X)[2] .== size(Y)[2]

    # Best Frob norm approximator
    Ã = Y*pinv(X)

    # Eigen Decomposition for solution
    Λ, W = eigen(Ã)

    if dt > 0.0
        # Casting Complex enforces results
        ω = log.(Complex.(Λ)) / dt
    else
        ω = []
    end

    return Koopman(Ã, Λ, ω, W, Y*X', X*X', alg, :PINV)
end

function koopman_svd(X::AbstractArray, Y::AbstractArray, alg::Symbol, dt::T1, dim::Int64 ,threshold::T2) where {T1 <: Real, T2 <: Real}
    # Compute the koopman operator based on svd
    U, S, V = svd(X)

    idx = iszero(threshold) ? collect(1:dim) : abs.(S) .>= threshold*maximum(S)

    # Operator approx
    Ã = U[:, idx]'*Y*V[:, idx]*Diagonal(one(eltype(S)) ./S[idx])

    # Eigen Decomposition for solution
    Λ, W = eigen(Ã)

    if dt > 0.0
        # Casting Complex enforces results
        ω = log.(Complex.(Λ)) / dt
    else
        ω = []
    end

    for (λ, wi) in zip(Λ, eachcol(W))
        wi = Y*V[:, idx]*Diagonal(one(eltype(S)) ./ S[idx])*wi/λ
    end

    # Compute the exact modes
    # W = Diagonal(eltype(Λ) ./ Λ)*Y*V[:, idx]*Diagonal(one(eltype(S)) ./S[idx])*W

    # Compute P and Q
    XX = Matrix(Diagonal(S[idx].^2))
    YX = U[:, idx]'*Y*V[:, idx]*Diagonal(S[idx])

    return Koopman(Ã, Λ, ω, W, YX, XX, alg, :SVD)
end

# Keep it simple
LinearAlgebra.eigen(m::Koopman) = m.λ, m.ϕ
LinearAlgebra.eigvals(m::Koopman) = m.λ
LinearAlgebra.eigvecs(m::Koopman) = m.ϕ

operator(m::Koopman) = m.Ã
modes(m::Koopman) = eigvecs(m)
frequencies(m::Koopman) =  !isempty(m.ω) ? m.ω : error("No continouos frequencies available.")
isstable(m::Koopman) = !isempty(m.ω) ? all(real.(frequencies(m)) .<= 0.0) : all(abs.(eigvals(m)) .<= 1)

iscontinouos(m::Koopman) = !isempty(m.ω) ? true : false

is_updateable(m::Koopman) = !(isnothing(m.Qₖ) && isnothing(m.Pₖ))

# TODO this can be done better, maybe use macros
function dynamics(m::Koopman; discrete::Bool = true)
    if discrete
    # Return an inline function
        @inline function dudt_(du, u, p, t)
            du .= m.Ã * u
        end
        return dudt_

    else
        @assert iscontinouos(m)
        A = m.ϕ*Diagonal(m.ω)*inv(m.ϕ)

        @inline function dudt_c(du, u, p, t)
            du .= A * u
        end

        return dudt_c
    end
end

function update!(m::Koopman, X::AbstractArray; dt::T1 = 0.0, threshold::T2 = 1e-3) where {T1 <: Real, T2 <: Real}
    update!(m, X[:, 1:end-1], X[:, 2:end], dt = dt, threshold = threshold)
end

# Update with new measurements
function update!(m::Koopman, x::AbstractArray, y::AbstractArray; dt::T1 = 0.0, threshold::T2 = 1e-3) where {T1 <: Real, T2 <: Real}
    @assert is_updateable(m) "Koopman operator can not be updated. Please provide sufficient matrices!"
    @assert size(x) == size(y)
    @assert size(x)[1] == size(operator(m))[2]

    # Check the error
    ϵ = norm(y - m.Ã*x, 2)

    if ϵ < threshold
        return
    end

    m.Qₖ += y*x'
    m.Pₖ += x*x'
    m.Ã = m.Qₖ*inv(m.Pₖ)
    m.λ, m.ϕ = eigen(m.Ã)

    if dt > 0.0
        # Casting Complex enforces results
        ω = log.(Complex.(m.λ)) / Δt
    else
        ω = []
    end
    return
end
