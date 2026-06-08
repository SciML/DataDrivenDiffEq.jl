using DataDrivenDiffEq
using DataDrivenSparse
using OrdinaryDiffEq
using StableRNGs
using Test
using StatsBase

function michaelis_menten(u, p, t)
    return [0.6 - 1.5u[1] / (0.3 + u[1])] # Should be 0.6*0.3+0.6u[1] - 1.5u[1] = u[2]*u[1]-0.3*u[2]
end

u0 = [0.5]
tspan = (0.0, 4.0)

problem_1 = ODEProblem(michaelis_menten, u0, tspan)
solution_1 = solve(problem_1, Tsit5(), saveat = 0.1)
problem_2 = ODEProblem(michaelis_menten, 2 * u0, tspan)
solution_2 = solve(problem_2, Tsit5(), saveat = 0.1)

@parameters t
@variables u[1:2]
u = collect(u)
h = [monomial_basis(u[1:1], 4)...]
basis = Basis([h; h .* u[2]], u[1:1], implicits = u[2:2])

prob = DataDrivenDataset(DataDrivenProblem(solution_1), DataDrivenProblem(solution_2))

@testset "Groundtruth" begin
    prob = DataDrivenDataset(DataDrivenProblem(solution_1), DataDrivenProblem(solution_2))

    opts = [
        ImplicitOptimizer(STLSQ(5.0e-2, 1.0e-7));
        ImplicitOptimizer(STLSQ(1.0e-2:1.0e-2:1.0e-1, 1.0e-7))
    ]
    for opt in opts
        res = solve(prob, basis, opt)
        @test r2(res) >= 0.9
        @test rss(res) < 1.0e-3
        @test dof(res) == 4
    end
end

@testset "Noise" begin
    rng = StableRNG(1111)
    prob = DataDrivenDataset(
        map((solution_1, solution_2)) do sol
            X = Array(sol)
            X .+= 0.01 * randn(rng, size(X))
            t = sol.t
            ContinuousDataDrivenProblem(X, t, GaussianKernel())
        end...
    )

    opts = [ImplicitOptimizer(ADMM(1.0e-2:1.0e-4:1.0e-1))]
    for opt in opts
        res = solve(prob, basis, opt, options = DataDrivenCommonOptions())
        @test r2(res) >= 0.9
        @test rss(res) <= 2.0e-1
        @test dof(res) == 3
    end
end
