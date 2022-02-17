using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra
using OrdinaryDiffEq

A = [-0.9 0.2; 0.0 -0.2]
B = [0.0; 1.0]
u0 = [10.0; -10.0]
tspan = (0.0, 10.0)

f(u,p,t) = A*u .+ B .* sin(0.5*t)

sys = ODEProblem(f, u0, tspan)
sol = solve(sys, Tsit5(), saveat = 0.05);

X = Array(sol)
t = sol.t
control(u,p,t) = [sin(0.5*t)]
prob = ContinuousDataDrivenProblem(X, t, U = control)

res = solve(prob, DMDSVD(), digits = 1)

system = result(res)

generator(system)

@parameters t
@variables x[1:2](t) u[1:1](t)

basis = Basis([x; u], x, controls = u, independent_variable = t, name = :LinearBasis)

sparse_res = solve(prob, basis, STLSQ(1e-1))

sparse_system = result(sparse_res)
println(sparse_system)

subs_control = (u[1] => sin(0.5*t))

eqs = map(equations(sparse_system)) do eq
    eq.lhs ~ substitute(eq.rhs, subs_control)
end

@named sys = ODESystem(
    eqs,
    get_iv(sparse_system),
    states(sparse_system),
    parameters(sparse_system)
    );

x0 = [x[1] => u0[1], x[2] => u0[2]]
ps = parameter_map(sparse_res)

ode_prob = ODEProblem(sys, x0, tspan, ps)
estimate = solve(ode_prob, Tsit5(), saveat = prob.t);

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

