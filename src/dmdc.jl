using OrdinaryDiffEq
using DataDrivenDiffEq
using LinearAlgebra
using Plots

mutable struct DMDc{K, B, Q, P}
    koopman::K
    forcing::B

    Qₖ::Q
    Pₖ::P
end


function DMDc(X::AbstractArray, Y::AbstractArray, Γ::AbstractArray; B::AbstractArray = [], Δt::Float64 = 0.0)
    @assert all(size(X) .== size(Y))
    @assert size(X)[2] .== size(Γ)[2]


    nₓ = size(X)[1]
    nᵤ = size(Γ)[1]

    if isempty(B)
        Ω = vcat(X, Γ)
        G = Y * pinv(Ω)

        Ã = G[:, 1:nₓ]
        B̃ = G[:, nₓ+1:end]
    else
        Ã = (Y - B*Γ)*pinv(X)
        B̃ = B
    end

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


function DMDc(X::AbstractArray, Γ::AbstractArray; B::AbstractArray = [], Δt::Float64 = 0.0)
    @assert size(X)[2]-1 == size(Γ)[2] "Provide consistent input data."
    return DMDc(X[:, 1:end-1], X[:, 2:end], Γ, B = B, Δt = Δt)
end

# Some nice functions
LinearAlgebra.eigen(m::DMDc) = eigen(m.koopman)
LinearAlgebra.eigvals(m::DMDc) = eigvals(m.koopman)
LinearAlgebra.eigvecs(m::DMDc) = eigvecs(m.koopman)

DataDrivenDiffEq.isstable(m::DMDc) = isstable(m.koopman)

get_dynamics(m::DMDc) = m.koopman
get_input_map(m::DMDc) = m.forcing

# TODO this can be done better, maybe use macros
function DataDrivenDiffEq.dynamics(m::DMDc; discrete::Bool = true)
    if discrete
        dims = size(m.forcing)
        nᵢ = length(dims)<=1 ? 1 : dims[2]
        println(zeros(eltype(m.forcing), nᵢ))
        function zero_callback(u, p, t)
            return zeros(eltype(m.forcing), nᵢ)
        end
        @inline function dudt_(u, p, t; y = zero_callback)
            m.koopman.Ã * u + m.forcing .* y(u, p, t)
        end
        return dudt_
    else
        throw(ErrorException("Continouos dynamics are not implemented right now."))
    end
end

X = [4 2 1 0.5 0.25; 7 0.7 0.07 0.007 0.0007]
U = [-4 -2 -1 -0.5]
B = Float32[1; 0]

sys = DMDc(X, U)
sys = DMDc(X, U, B = B)

isstable(sys)

get_input_map(sys)

get_dynamics(sys)

eigen(sys)

dudt_ = dynamics(sys)
dudt_(X[:, 1], [], 0.0)

prob = DiscreteProblem(dudt_, X[:, 1], (0.0, 10.0))
sol = solve(prob)

plot(sol)
