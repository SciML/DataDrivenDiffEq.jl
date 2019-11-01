import LinearAlgebra: eigen
import LinearAlgebra: eigvals, eigvecs


mutable struct ExtendedDMD{D,O,C, F} <: abstractKoopmanOperator
    koopman::D
    output::O
    basis::C
    f::F # This is the placeholder for the equation
end

# Make the struct callable for transformations
(m::ExtendedDMD)(u; p = []) = m.f(u, p)

# Some nice functions
eigen(m::ExtendedDMD) = eigen(m.koopman)
eigvals(m::ExtendedDMD) = eigvals(m.koopman)
eigvecs(m::ExtendedDMD) = eigvecs(m.koopman)

function ExtendedDMD(X::AbstractArray, Ψ; p::AbstractArray = [],  B::AbstractArray = reshape([], 0,0), Δt::Float64 = 1.0)
    return ExtendedDMD(X[:, 1:end-1], X[:, 2:end], Ψ, p = p, B = B, Δt = Δt)
end

function ExtendedDMD(X::AbstractArray, Y::AbstractArray, Ψ; p::AbstractArray = [], B::AbstractArray = reshape([], 0,0), Δt::Float64 = 1.0)
    @assert size(X)[2] .== size(Y)[2]
    @assert size(Y)[1] .<= size(Y)[2]

    # Based upon William et.al. , A Data-Driven Approximation of the Koopman operator

    # Number of states and measurements
    N,M = size(X)

    # Convert the modeling toolkit vector of operations into a function
    f, _ = ModelingToolkit.build_function(Ψ, [ModelingToolkit.vars(Ψ)...], [], (), ModelingToolkit.simplified_expr, Val{false})

    # Compute the transformed data
    Ψ₀ = hcat([f(xi, p) for xi in eachcol(X)]...)
    Ψ₁ = hcat([f(xi, p) for xi in eachcol(Y)]...)
    Op = ExactDMD(Ψ₀, Ψ₁) # Initial guess based upon the basis

    # Transform back to states
    if isempty(B)
        B = X*pinv(Ψ₀)
    end

    # TODO Maybe reduce the observable space here
    return ExtendedDMD(Op, B, Ψ, f)
end


# TODO This is not tested and will most likely fail when used with
# singular basis
function update!(m::ExtendedDMD, x::AbstractArray, y::AbstractArray; p::AbstractArray = [], Δt::Float64 = 0.0)
    Ψ₀ = m.f(x, p)
    Ψ₁ = m.f(y, p)
    update!(m.koopman, Ψ₀, Ψ₁, Δt = Δt)
    return
end

# TODO How to implement continouos time dynamics?
# We would need ∂Ψ/∂x or ∂Ψ/∂t

function dynamics(m::ExtendedDMD)
    # Create a set of nonlinear eqs
    p_ = m.output*m.koopman.Ã
    function dudt_(du, u, p, t)
        du .= p_*m.f(u, p)
    end
end


function linear_dynamics(m::ExtendedDMD; discrete::Bool = true)
    return dynamics(m.koopman, discrete = discrete)
end

# Reduction for basis
function basis(m::ExtendedDMD; threshold = 1e-5)
    b = approximator.output*approximator.koopman.Ã
    b[abs.(b) .< threshold] .= 0.0
    return simplified_expr.(simplify_constants.(b*approximator.basis))
end




#function reduce_basis!(m::ExtendedDMD, threshold::Number = 1e-3)
#    b = approximator.output*approximator.koopman.Ã
#    inds = sum(abs.(b), dims = 1) .> threshold
#
#    m.basis = m.basis[vec(inds)]
#    return
#end
