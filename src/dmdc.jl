using DifferentialEquations
using Plots
using DataDrivenDiffEq
using LinearAlgebra

mutable struct DMDc{K, B, Q, P}
    koopman::K
    forcing::B

    Qₖ::Q
    Pₖ::P
end


function DMDc(X::AbstractArray, Y::AbstractArray, Γ::AbstractArray; Δt::Float64 = 0.0)
    @assert all(size(X) .== size(Y))
    @assert size(X)[2] .== size(Γ)[2]


    nₓ = size(X)[1]
    nᵤ = size(Γ)[1]

    Ω = vcat(X, Γ)
    G = X * pinv(Ω)

    Ã = G[:, 1:nₓ]
    B̃ = G[:, nₓ+1:end]

    # Eigen Decomposition for solution
    Λ, W = eigen(Ã)

    if Δt > 0.0
        # Casting Complex enforces results
        ω = log.(Complex.(Λ)) / Δt
    else
        ω = []
    end

    koopman = ExactDMD(Ã, Λ, ω, W, nothing, nothing)


    return DMDc(koopman, B̃, nothing, nothing)
end

function DMDc(X::AbstractArray, Γ::AbstractArray; Δt::Float64 = 0.0)
    @assert size(X)[2]-1 == size(Γ)[2] "Provide consistent input data."
    return DMDc(X[:, 1:end-1], X[:, 2:end], Γ, Δt = Δt)
end

# Some nice functions
LinearAlgebra.eigen(m::DMDc) = eigen(m.koopman)
LinearAlgebra.eigvals(m::DMDc) = eigvals(m.koopman)
LinearAlgebra.eigvecs(m::DMDc) = eigvecs(m.koopman)

get_dynamics(m::DMDc) = m.koopman
get_input_map(m::DMDc) = m.forcing

# TODO this can be done better, maybe use macros
function DataDrivenDiffEq.dynamics(m::DMDc; discrete::Bool = true)
    if discrete
        nᵢ = size(m.forcing)[2]
        function zero_callback(u, p, t)
            return zeros(eltype(m.forcing), nᵢ)
        end
        @inline function dudt_(du, u, p, t; y = zero_callback)
            du .= m.koopman.Ã * u + m.forcing * y(u, p, t)
        end
        return dudt_
    else
        throw(ErrorException("Continouos dynamics are not implemented right now."))
    end
end
