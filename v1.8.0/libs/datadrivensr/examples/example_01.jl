using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenSR

A = [-0.9 0.2; 0.0 -0.5]
B = [0.0; 1.0]
u0 = [10.0; -10.0]
tspan = (0.0, 20.0)

f(u, p, t) = A * u .+ B .* sin(0.5 * t)

sys = ODEProblem(f, u0, tspan)
sol = solve(sys, Tsit5(), saveat = 0.01);

X = Array(sol)
t = sol.t
U = permutedims(sin.(0.5 * t))
prob = ContinuousDataDrivenProblem(X, t, U = U)

eqsearch_options = SymbolicRegression.Options(binary_operators = [+, *],
    loss = L1DistLoss(),
    verbosity = -1, progress = false, npop = 30,
    timeout_in_seconds = 60.0)

alg = EQSearch(eq_options = eqsearch_options)

res = solve(prob, alg, options = DataDrivenCommonOptions(maxiters = 100))

loglikelihood(res)

system = get_basis(res)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
