using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenDMD

function slow_manifold(du, u, p, t)
    du[1] = p[1] * u[1]
    du[2] = p[2] * (u[2] - u[1]^2)
end

u0 = [3.0; -2.0]
tspan = (0.0, 5.0)
p = [-0.8; -0.7]

problem = ODEProblem{true, SciMLBase.NoSpecialize}(slow_manifold, u0, tspan, p)
solution = solve(problem, Tsit5(), saveat = 0.1);

prob = DataDrivenProblem(solution)

@parameters t
@variables u(t)[1:2]
Ψ = Basis([u; u[1]^2], u, independent_variable = t)
res = solve(prob, Ψ, DMDPINV(), digits = 2)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

