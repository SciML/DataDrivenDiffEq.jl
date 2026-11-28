using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenSR

function pendulum!(du, u, p, t)
    du[1] = u[2]
    du[2] = -9.81 * sin(u[1])
end

u0 = [0.1, π / 2]
tspan = (0.0, 10.0)
sys = ODEProblem{true, SciMLBase.NoSpecialize}(pendulum!, u0, tspan)
sol = solve(sys, Tsit5());

prob = DataDrivenProblem(sol)

@variables u[1:2]
u = collect(u)

basis = Basis([polynomial_basis(u, 2); sin.(u)], u)

eqsearch_options = SymbolicRegression.Options(binary_operators = [+, *],
                                              loss = L1DistLoss(),
                                              verbosity = -1, progress = false, npop = 30,
                                              timeout_in_seconds = 60.0)

alg = EQSearch(eq_options = eqsearch_options)

res = solve(prob, basis, alg, options = DataDrivenCommonOptions(maxiters = 100))

system = get_basis(res)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

