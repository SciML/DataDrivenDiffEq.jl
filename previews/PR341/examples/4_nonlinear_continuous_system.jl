using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using OrdinaryDiffEq

function slow_manifold(du, u, p, t)
    du[1] = p[1] * u[1]
    du[2] = p[2] * (u[2] - u[1]^2)
end

u0 = [3.0; -2.0]
tspan = (0.0, 5.0)
p = [-0.8; -0.7]

problem = ODEProblem(slow_manifold, u0, tspan, p)
solution = solve(problem, Tsit5(), saveat = 0.01)

prob = ContinuousDataDrivenProblem(solution)

@parameters t
@variables u[1:2](t)
Ψ = Basis([u; u[1]^2], u, independent_variable = t)
res = solve(prob, Ψ, DMDPINV(), digits = 1)
system = result(res)

sparse_res = solve(prob, Ψ, STLSQ(), digits = 1)

sparse_system = result(sparse_res)

parameter_map(res)

parameter_map(sparse_res)

@named sys = ODESystem(
    equations(sparse_system),
    get_iv(sparse_system),
    states(sparse_system),
    parameters(sparse_system)
    );


x0 = [u[1] => u0[1], u[2] => u0[2]]
ps = parameter_map(sparse_res)

ode_prob = ODEProblem(sys, x0, tspan, ps)
estimate = solve(ode_prob, Tsit5(), saveat = prob.t);

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

