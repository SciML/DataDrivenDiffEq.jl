mutable struct Koopman{M,L,W,F, Q, P} <: abstractKoopmanOperator
    Ã::M # Approximation of the operator
    λ::L # Eigenvalues (discrete time)
    ω::W # Frequencies
    ϕ::F # Modes


    # For online DMD
    Qₖ::Q
    Pₖ::P

    # Algorithm
    alg::Symbol
    method::Symbol
end

function Koopman(Ã::AbstractArray, alg::Symbol; dt::T = 0.0, method::Symbol = :PINV) where T <: Real
    # Eigen Decomposition for solution
    Λ, W = eigen(Ã)

    if Δt > 0.0
        # Casting Complex enforces results
        ω = log.(Complex.(Λ)) / Δt
    else
        ω = []
    end

    return Koopman(Ã, Λ, ω, W, nothing, nothing, alg, :PINV)
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

    if Δt > 0.0
        # Casting Complex enforces results
        ω = log.(Complex.(m.λ)) / Δt
    else
        ω = []
    end
    return
end
