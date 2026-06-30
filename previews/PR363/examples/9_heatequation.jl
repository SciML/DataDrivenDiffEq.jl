using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using OrdinaryDiffEq
using DiffEqOperators

u_analytic(x, t) = sin(2 * π * x) * exp(-t * (2 * π)^2)
nknots = 100
h = 1.0 / (nknots + 1)
knots = range(h, step = h, length = nknots)
ord_deriv = 2
ord_approx = 2

const bc = Dirichlet0BC(Float64)
const Δ = CenteredDifference(ord_deriv, ord_approx, h, nknots)

t0 = 0.0
t1 = 1.0
u0 = u_analytic.(knots, t0)

step(u, p, t) = Δ * bc * u
prob = ODEProblem(step, u0, (t0, t1))
alg = KenCarp4()
de_solution = solve(prob, alg);

∂U = reduce(vcat, map(1:4) do n
                δ = CenteredDifference(n, ord_approx, h, nknots)
                reshape(δ * bc * Array(de_solution), 1, prod(size(de_solution)))
            end)

U = reshape(Array(de_solution), 1, prod(size(de_solution)))
∂ₜU = reshape(Array(de_solution(de_solution.t, Val{1})), 1, prod(size(de_solution)))

problem = DataDrivenProblem(U, DX = ∂ₜU, U = ∂U)

@parameters t x
@variables u(x, t)
∂u = map(1:4) do n
    d = Differential(x)^n
    d(u)
end

basis = Basis([monomial_basis([u], 5); ∂u], [u], independent_variable = t, controls = ∂u);
println(basis) #hide

sampler = DataSampler(Split(ratio = 0.8), Batcher(n = 10))
solution = solve(problem, basis, STLSQ(1e-2:1e-2:5e-1), sampler = sampler, by = :best)

result(solution)
println(result(solution)) #hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

