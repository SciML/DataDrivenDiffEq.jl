using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra
using StatsBase

@variables x[1:3]
@parameters p[1:2]

eqs = [sin(p[1]*x[1])+x[2]; p[2]*x[3]]

b = Basis(eqs, x, parameters = p)

x = randn(3, 100)
p0 = randn(2)
y = b(x, p0)
ŷ = y .+ 0.01*randn(size(y))
prob = DirectDataDrivenProblem(x, y, p = p0)

sol = DataDrivenSolution(b, prob)
@test dof(sol) == 2
@test rss(sol) == 0
@test aic(sol) == -Inf
@test bic(sol) == -Inf
@test aicc(sol) == -Inf
@test loglikelihood(sol) == Inf
@test r2(sol) ≈ 1.0
@test nobs(sol) == prod(size(y))
@test_nowarn summarystats(sol)

@test get_problem(sol) == prob
@test get_result(sol) == DataDrivenDiffEq.ErrorDataDrivenResult()
@test get_algorithm(sol) == DataDrivenDiffEq.ZeroDataDrivenAlgorithm()
@test isa(get_basis(sol), DataDrivenDiffEq.AbstractBasis)
@test !is_converged(sol)

ŷ = y .+ 0.01*randn(size(y))
prob = DirectDataDrivenProblem(x, ŷ, p = p0)
sol_2 = DataDrivenSolution(b, prob)
@test aic(sol_2) <= -1800.
@test bic(sol_2) <= -1800.
@test aicc(sol_2) <= -1800.
@test 900.0 <= loglikelihood(sol_2) <= 1000.0
@test r2(sol_2) ≈ 0.9998 atol = 1e-3
@test nobs(sol_2) == prod(size(y))
@test_nowarn summarystats(sol_2)
