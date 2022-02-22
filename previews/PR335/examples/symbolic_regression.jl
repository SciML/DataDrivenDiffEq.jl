#note # Symbolic regression is using  regularized evolution, simulated annealing, and gradient-free optimization to find suitable equations. Hence, the performance might differ and depends strongly on the hyperparameters of the optimization. This example might not recover the groundtruth, but is showing off the use within `DataDrivenDiffEq.jl`.

using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra
using OrdinaryDiffEq
using SymbolicRegression

A = [-0.9 0.2; 0.0 -0.5]
B = [0.0; 1.0]
u0 = [10.0; -10.0]
tspan = (0.0, 20.0)

f(u,p,t) = A*u .+ B .* sin(0.5*t)

sys = ODEProblem(f, u0, tspan)
sol = solve(sys, Tsit5(), saveat = 0.01);

X = Array(sol)
t = sol.t
U = permutedims(sin.(0.5*t))
prob = ContinuousDataDrivenProblem(X, t, U = U)

alg = EQSearch([-, *], loss = L1DistLoss(), verbosity = 0, maxsize = 9, batching = true, batchSize = 50, parsimony = 0.01f0)

res = solve(prob, alg, max_iter = 300, numprocs = 0, multithreading = false)

system = result(res)
println(system)

u = controls(system)
t = get_iv(system)

subs_control = (u[1] => sin(0.5*t))

eqs = map(equations(system)) do eq
    eq.lhs ~ substitute(eq.rhs, subs_control)
end

@named sys = ODESystem(
    eqs,
    get_iv(system),
    states(system),
    []
    );

x = states(system)
x0 = [x[1] => u0[1], x[2] => u0[2]]

ode_prob = ODEProblem(sys, x0, tspan)
estimate = solve(ode_prob, Tsit5(), saveat = prob.t);

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

