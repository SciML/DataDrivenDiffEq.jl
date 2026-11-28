using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenDMD

A = [-0.9 0.2; 0.0 -0.2]
u0 = [10.0; -10.0]
tspan = (0.0, 10.0)

f(u, p, t) = A * u

sys = ODEProblem(f, u0, tspan)
sol = solve(sys, Tsit5(), saveat = 0.05);

X = Array(sol)
t = sol.t
prob = ContinuousDataDrivenProblem(X, t)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
