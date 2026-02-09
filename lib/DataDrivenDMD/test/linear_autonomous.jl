using DataDrivenDiffEq
using DataDrivenDMD
using LinearAlgebra
using Test
using StatsBase
using StableRNGs
using OrdinaryDiffEq

rng = StableRNG(42)

@testset "Linear Discrete System" begin
    # Create some linear data
    A = [0.9 -0.2; 0.0 0.2]
    y = [[10.0; -10.0]]
    for i in 1:10
        push!(y, A * y[end])
    end
    X = hcat(y...)

    @testset "Groundtruth" begin
        prob = DiscreteDataDrivenProblem(X, t = 1:11)

        for alg in [DMDPINV(), DMDSVD(), TOTALDMD()]
            res = solve(prob, alg)
            @test rss(res) <= 1.0e-2
            @test r2(res) ≈ 0.95 atol = 5.0e-1
            @test dof(res) == 3
            @test loglikelihood(res) >= 400.0

            foreach(get_results(res)) do operator_res
                @test Matrix(get_operator(operator_res)) ≈ A
                @test isempty(get_inputmap(operator_res))
                @test get_outputmap(operator_res) ≈ I(2)
                @test rss(operator_res) <= 1.0e-10
            end
        end
    end

    @testset "Noise" begin
        Xₙ = X .+ 0.01 * randn(rng, size(X))
        prob = DiscreteDataDrivenProblem(Xₙ, t = 1:11)

        for alg in [DMDPINV(), DMDSVD(), TOTALDMD()]
            res = solve(prob, alg)
            @test rss(res) <= 1.0e-2
            @test r2(res) ≈ 1.0 atol = 1.0e-1
            @test dof(res) == 4
            @test loglikelihood(res) >= 85.0

            foreach(get_results(res)) do operator_res
                @test Matrix(get_operator(operator_res)) ≈ A atol = 1.0e-2
                @test isempty(get_inputmap(operator_res))
                @test get_outputmap(operator_res) ≈ I(2)
                @test rss(operator_res) <= 1.0e-2
            end
        end
    end
end

@testset "Linear Continuous System" begin
    A = [-0.9 0.1; 0.0 -0.2]
    f(u, p, t) = A * u
    u0 = [10.0; -20.0]
    prob = ODEProblem(f, u0, (0.0, 10.0))
    sol = solve(prob, Tsit5(), saveat = 0.001)

    @testset "Groundtruth" begin
        prob = DataDrivenProblem(sol)

        for alg in [DMDPINV(), DMDSVD(), TOTALDMD()]
            res = solve(prob, alg)
            @test rss(res) <= 1.0e-2
            @test r2(res) ≈ 0.95 atol = 5.0e-1
            @test dof(res) == 3
            @test loglikelihood(res) >= 400.0e3

            foreach(get_results(res)) do operator_res
                @test Matrix(get_operator(operator_res)) ≈ A
                @test isempty(get_inputmap(operator_res))
                @test get_outputmap(operator_res) ≈ I(2)
                @test rss(operator_res) <= 1.0e-10
            end
        end
    end

    @testset "Noise" begin
        X = Array(sol)
        X .+= 0.01 * randn(rng, size(X))
        t = sol.t
        prob = ContinuousDataDrivenProblem(X, t, GaussianKernel())

        for alg in [DMDPINV(), DMDSVD(), TOTALDMD()]
            res = solve(prob, alg)
            @test rss(res) <= 2.0
            @test r2(res) ≈ 1.0 atol = 1.0e-2
            @test dof(res) == 4
            @test loglikelihood(res) >= 85.0e3
        end
    end
end

@testset "Low Rank Continuous System" begin
    K̃ = -0.5 * I + [0 0 -0.2; 0.1 0 -0.1; 0.0 -0.2 0]
    F = qr(randn(rng, 20, 3))
    Q = F.Q[:, 1:3]
    dudt(u, p, t) = K̃ * u
    prob = ODEProblem(dudt, [10.0; 0.3; -5.0], (0.0, 10.0))
    sol_ = solve(prob, Tsit5(), saveat = 0.01)

    # True Rank is 3
    X = Q * Array(sol_) + 1.0e-3 * randn(rng, 20, 1001)
    DX = Q * Array(sol_(sol_.t, Val{1})) + 1.0e-3 * randn(rng, 20, 1001)
    ddprob = ContinuousDataDrivenProblem(X, sol_.t, DX = DX)

    for alg in [TOTALDMD(3, DMDPINV()); TOTALDMD(0.01, DMDSVD(3))]
        res = solve(ddprob, alg, digits = 2)
        @test rss(res) <= 1.0e-1
        @test r2(res) ≈ 1.0 atol = 1.0e-2
        @test dof(res) == 400
        @test loglikelihood(res) >= 99.0e3

        foreach(get_results(res)) do operator_res
            K = Matrix(get_operator(operator_res))
            @test Q' * K * Q ≈ K̃ atol = 1.0e-1
            @test Q * K̃ * Q' ≈ K atol = 1.0e-1
        end
    end
end
