
function ExtendedDMD(X::AbstractArray, Ψ::abstractBasis; p::AbstractArray = [],  B::AbstractArray = [])
    return ExtendedDMD(X[:, 1:end-1], X[:, 2:end], Ψ, p = p, B = B)
end

function ExtendedDMD(X::AbstractArray, Y::AbstractArray, Ψ::abstractBasis; p::AbstractArray = [], B::AbstractArray = [])
    @assert size(X)[2] .== size(Y)[2] "Provide consistent dimensions for data"
    @assert size(Y)[1] .<= size(Y)[2] "Provide consistent dimensions for data"

    # Based upon William et.al. , A Data-Driven Approximation of the Koopman operator

    # Number of states and measurements
    N,M = size(X)

    # Compute the transformed data
    Ψ₀ = Ψ(X, p)
    Ψ₁ = Ψ(Y, p)


    Op = ExactDMD(Ψ₀, Ψ₁, dt = dt) # Initial guess based upon the basis

    # Transform back to states
    if isempty(B)
        B = X*pinv(Ψ₀)
    end

    # TODO Maybe reduce the observable space here
    return ExtendedDMD(Op, B, Ψ)
end

function update!(m::ExtendedDMD, x::AbstractArray, y::AbstractArray; p::AbstractArray = [], dt::T = 0.0, threshold::Float64 = 1e-3) where T <: Real
    Ψ₀ = m.basis(x, p)
    Ψ₁ = m.basis(y, p)
    update!(m.koopman, Ψ₀, Ψ₁, dt = dt, threshold = threshold)
    return
end

# TODO How to implement continouos time dynamics?
# We would need ∂Ψ/∂x or ∂Ψ/∂t
function dynamics(m::ExtendedDMD)
    # Create a set of nonlinear eqs
    p_ = m.output*m.koopman.Ã
    function dudt_(du, u, p, t)
        mul!(du,p_,m.basis(u, p, t))
    end
end


function linear_dynamics(m::ExtendedDMD; discrete::Bool = true)
    return dynamics(m.koopman, discrete = discrete)
end

# Reduction for basis
# This is a fairly naive approach of doing this
function reduce_basis(m::ExtendedDMD; threshold = 1e-5)
    b = m.output*m.koopman.Ã
    inds = sum(abs, b, dims = 1) .> threshold
    return Basis(m.basis.basis[vec(inds)], variables(m.basis), parameters = parameters(m.basis))
end