import LinearAlgebra: eigen
import LinearAlgebra: eigvals, eigvecs


mutable struct ExtendedDMD{D,O,C} <: abstractKoopmanOperator
    koopman::D
    output::O
    basis::C
end

# Make the struct callable for transformations
(m::ExtendedDMD)(u; p = []) = m.basis(u, p = p)

# Some nice functions
LinearAlgebra.eigen(m::ExtendedDMD) = eigen(m.koopman)
LinearAlgebra.eigvals(m::ExtendedDMD) = eigvals(m.koopman)
LinearAlgebra.eigvecs(m::ExtendedDMD) = eigvecs(m.koopman)

function ExtendedDMD(X::AbstractArray, Ψ::abstractBasis; p::AbstractArray = [],  B::AbstractArray = reshape([], 0,0), dt::T = 0.0) where T <: Real
    return ExtendedDMD(X[:, 1:end-1], X[:, 2:end], Ψ, p = p, B = B, dt = dt)
end

function ExtendedDMD(X::AbstractArray, Y::AbstractArray, Ψ::abstractBasis; p::AbstractArray = [], B::AbstractArray = reshape([], 0,0), dt::T = 0.0) where T <: Real
    # Based upon William et.al. , A Data-Driven Approximation of the Koopman operator

    # Number of states and measurements
    N,M = size(X)

    # Compute the transformed data
    Ψ₀ = hcat([Ψ(xi, p = p) for xi in eachcol(X)]...)
    Ψ₁ = hcat([Ψ(xi, p = p) for xi in eachcol(Y)]...)
    Op = ExactDMD(Ψ₀, Ψ₁, dt = dt) # Initial guess based upon the basis

    # Transform back to states
    if isempty(B)
        B = X*pinv(Ψ₀)
    end

    # TODO Maybe reduce the observable space here
    return ExtendedDMD(Op, B, Ψ)
end

function update!(m::ExtendedDMD, x::AbstractArray, y::AbstractArray; p::AbstractArray = [], dt::T1 = 0.0, threshold::T2 = 1e-3) where {T1 <: Real, T2 <: Real}
    Ψ₀ = m.basis(x, p = p)
    Ψ₁ = m.basis(y, p = p)
    update!(m.koopman, Ψ₀, Ψ₁, Δt = Δt, threshold = threshold)
    return
end

# We can provide the dynamics like in "normal" DMD
# since this
function dynamics(m::ExtendedDMD)
    # Create a set of nonlinear eqs
    p_ = m.output*m.koopman.Ã
    function dudt_(du, u, p, t)
        du .= p_*m.basis(u, p = p)
    end
end


function linear_dynamics(m::ExtendedDMD; discrete::Bool = true)
    return dynamics(m.koopman, discrete = discrete)
end

# Reduction for basis
# This is a fairly naive approach of doing this
function reduce_basis(m::ExtendedDMD; threshold::T = 1e-5) where T <: Real
    @assert threshold > 0 "Threshold has to be positive definite!"
    b = m.output*m.koopman.Ã
    inds = sum(abs, b, dims = 1) .> threshold
    return Basis(m.basis.basis[vec(inds)], variables(m.basis), parameters = parameters(m.basis))
end
