using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenSparse

function michaelis_menten(u, p, t)
    [0.6 - 1.5u[1]/(0.3+u[1])]
end

u0 = [0.5]

ode_problem = ODEProblem(michaelis_menten, u0, (0.0, 4.0));

prob = DataDrivenDataset(map(1:2) do i
    solve(
        remake(ode_problem, u0 = i*u0),
        Tsit5(), saveat = 0.1, tspan = (0.0, 4.0)
    )
end...)

@parameters t
@variables u(t)[1:1]
u = collect(u)
D = Differential(t)
h = [monomial_basis(u[1:1], 4)...]
basis = Basis([h; h .* (D(u[1]))], u, implicits = D.(u), iv = t)

opt = ImplicitOptimizer(1e-1:1e-1:5e-1)
res = solve(prob, basis, opt)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

