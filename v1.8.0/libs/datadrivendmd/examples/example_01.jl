using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenDMD

A = [0.9 -0.2; 0.0 0.2]
u0 = [10.0; -10.0]
tspan = (0.0, 11.0)

f(u, p, t) = A * u

sys = DiscreteProblem(f, u0, tspan)
sol = solve(sys, FunctionMap());

prob = DataDrivenProblem(sol)

res = solve(prob, DMDSVD(), digits = 1)

get_basis(res)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
