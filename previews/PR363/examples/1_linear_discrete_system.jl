using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra
using OrdinaryDiffEq

A = [0.9 -0.2; 0.0 0.2]
u0 = [10.0; -10.0]
tspan = (0.0, 11.0)

f(u, p, t) = A * u

sys = DiscreteProblem(f, u0, tspan)
sol = solve(sys, FunctionMap());

prob = DataDrivenProblem(sol)

res = solve(prob, DMDSVD(), digits = 1)

system = result(res)
using Symbolics

using ModelingToolkit

@parameters t
@variables x[1:2](t)

basis = Basis(x, x, independent_variable = t, name = :LinearBasis)

sparse_res = solve(prob, basis, STLSQ())

sparse_system = result(sparse_res)

@named sys = DiscreteSystem(equations(sparse_system), get_iv(sparse_system),
                            states(sparse_system), parameters(sparse_system))

x0 = [x[1] => u0[1], x[2] => u0[2]]
ps = parameter_map(sparse_res)

discrete_prob = DiscreteProblem(sys, x0, tspan, ps)
estimate = solve(discrete_prob, FunctionMap());

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

