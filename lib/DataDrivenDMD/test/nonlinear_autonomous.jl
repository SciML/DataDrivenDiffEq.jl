using DataDrivenDiffEq
using DataDrivenDMD
using LinearAlgebra
using Test
using StatsBase
using StableRNGs
using OrdinaryDiffEq

rng = StableRNG(42)

@testset "Slow Manifold System" begin
    function slow_manifold(du, u, p, t)
        du[1] = p[1] * u[1]
        du[2] = p[2] * (u[2] - u[1]^2)
    end

    u0 = [3.0; -2.0]
    tspan = (0.0, 5.0)
    p = [-0.8; -0.7]

    problem = ODEProblem(slow_manifold, u0, tspan, p)
    solution = solve(problem, Tsit5(), saveat = 0.01)

    @testset "Groundtruth" begin
        prob = DataDrivenProblem(solution, use_interpolation = true)

        @variables u[1:2]
        Ψ = Basis([u; u[1]^2], u)

        for alg in [DMDPINV(); DMDSVD(); TOTALDMD()]
            res = solve(prob, Ψ, alg, options = DataDrivenCommonOptions(digits = 1))
            @test get_parameter_values(res.basis) ≈ [-0.8, -0.7, 0.6]
            @test loglikelihood(res) >= 1700.0
            @test r2(res) >= 0.98
        end
    end
end

@testset "Slow Manifold Discrete System" begin
    function slow_manifold(du, u, p, t)
        du[1] = p[1] * u[1]
        du[2] = p[2] * u[2] + (p[1]^2 - p[2]) * u[1]^2
    end

    u0 = [3.0; -2.0]
    tspan = (0.0, 10.0)
    p = [-0.8; -0.7]

    problem = DiscreteProblem(slow_manifold, u0, tspan, p)
    solution = solve(problem, FunctionMap())

    prob = DataDrivenProblem(solution)

    @variables u[1:2]

    Ψ = Basis([u[1]; u[1]^2; u[2] - u[1]^2], u)

    for alg in [DMDPINV(); DMDSVD(); TOTALDMD()]
        res = solve(prob, Ψ, alg, options = DataDrivenCommonOptions(digits = 2))
        @test get_parameter_values(res.basis)≈[-0.8, 0.63, -0.7] atol=5e-2
        @test loglikelihood(res) >= 50.0
        @test r2(res) >= 0.98
        @test rss(res) <= 1e-1
    end
end
