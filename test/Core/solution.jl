using DataDrivenDiffEq
using LinearAlgebra
using StatsBase
using Random
using DataDrivenDiffEq.CommonSolve

rng = Random.default_rng()
Random.seed!(0)

p0 = randn(rng, 2)
@variables x[1:3]
@parameters p₁ = p0[1] p₂ = p0[2]
p = [p₁; p₂]
eqs = [sin(p[1] * x[1]) + x[2]; p[2] * x[3]]

b = Basis(eqs, x, parameters = p)
@test get_parameter_values(b) == p0
x = randn(rng, 3, 100)
t = collect(0.0:1.0:99.0)
y = b(x, p0, t)
ŷ = y .+ 0.01 * randn(rng, size(y))

struct DummyDataDrivenAlgorithm <: DataDrivenDiffEq.AbstractDataDrivenAlgorithm end
struct DummyDataDrivenResult{IP} <: DataDrivenDiffEq.AbstractDataDrivenResult
    internal::IP
end

function CommonSolve.solve!(
        p::DataDrivenDiffEq.InternalDataDrivenProblem{
            DummyDataDrivenAlgorithm,
        }
    )
    return DummyDataDrivenResult(p)
end

prob = DirectDataDrivenProblem(x, y, p = p0)
dummy_sol = solve(prob, b, DummyDataDrivenAlgorithm())
internal_problem = dummy_sol.internal
sol = DataDrivenSolution(
    b, prob, DataDrivenDiffEq.ZeroDataDrivenAlgorithm(),
    DataDrivenDiffEq.AbstractDataDrivenResult[dummy_sol],
    internal_problem
)

@test dof(sol) == 2
@test rss(sol) == 0
@test aic(sol) == -Inf
@test bic(sol) == -Inf
@test aicc(sol) == -Inf
@test loglikelihood(sol) == Inf
@test r2(sol) ≈ 1.0
@test nobs(sol) == prod(size(y))
@test_nowarn summarystats(sol)
# We remake the problem now
@test get_problem(sol) != prob
@test get_results(sol) == [dummy_sol]
@test get_algorithm(sol) == DataDrivenDiffEq.ZeroDataDrivenAlgorithm()
@test isa(get_basis(sol), DataDrivenDiffEq.AbstractBasis)
@test !is_converged(sol)

ŷ = y .+ 0.01 * randn(rng, size(y))
prob = DirectDataDrivenProblem(x, ŷ, p = p0)
dummy_sol = solve(prob, b, DummyDataDrivenAlgorithm())
internal_problem = dummy_sol.internal
sol_2 = DataDrivenSolution(
    b, prob, DataDrivenDiffEq.ZeroDataDrivenAlgorithm(),
    DataDrivenDiffEq.AbstractDataDrivenResult[dummy_sol],
    internal_problem
)

@test aic(sol_2) <= -1800.0
@test bic(sol_2) <= -1700.0
@test aicc(sol_2) <= -1800.0
@test 900.0 <= loglikelihood(sol_2) <= 1000.0
@test r2(sol_2) ≈ 0.99998 atol = 1.0e-3
@test nobs(sol_2) == prod(size(y))
@test_nowarn summarystats(sol_2)
