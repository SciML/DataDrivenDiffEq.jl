mutable struct ExtendedDMD{D,O,C} <: abstractKoopmanOperator
    koopman::D
    output::O
    basis::C
end

function ExtendedDMD(X::AbstractArray, Ψ::BasisCandidate; B::AbstractArray = reshape([], 0,0), Δt::Float64 = 1.0)
    return ExtendedDMD(X[:, 1:end-1], X[:, 2:end], Ψ, B = B, Δt = Δt)
end

function ExtendedDMD(X::AbstractArray, Y::AbstractArray, Ψ::BasisCandidate; B::AbstractArray = reshape([], 0,0), Δt::Float64 = 1.0)
    @assert size(X)[2] .== size(Y)[2]
    @assert size(Y)[1] .<= size(Y)[2]

    # Based upon William et.al. , A Data-Driven Approximation of the Koopman operator

    # Number of states and measurements
    N,M = size(X)

    # Compute the transformed data
    Ψ₀ = evaluate(Ψ, X)
    Ψ₁ = evaluate(Ψ, Y)
    Op = ExactDMD(Ψ₀, Ψ₁) # Initial guess based upon the basis

    # Transform back to states
    if isempty(B)
        B = X*pinv(Ψ₀)
    end

    # TODO Maybe reduce the observable space here

    return ExtendedDMD(Op, B, Ψ)
end

# TODO This is not tested and will most likely fail when used with
# singular basis
function update!(m::ExtendedDMD, x::AbstractArray, y::AbstractArray; Δt::Float64 = 0.0)
    Ψ₀ = m.basis(x)
    Ψ₁ = m.basis(y)
    update!(m.koopman, Ψ₀, Ψ₁, Δt = Δt)
    return
end

# TODO How to implement continouos time dynamics?
# We would need ∂Ψ/∂x or ∂Ψ/∂t
function dynamics(m::ExtendedDMD)
    # Create a set of nonlinear eqs
    Ψᵣ, p_ = collapse(m.basis, m.output*m.koopman.Ã)
    function dudt_(du, u, p, t)
        du .= p_*Ψᵣ(u)
    end
end

function linear_dynamics(m::ExtendedDMD; discrete::Bool = true)
    return dynamics(m.koopman, discrete = discrete)
end
