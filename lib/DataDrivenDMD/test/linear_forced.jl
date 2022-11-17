using DataDrivenDiffEq
using DataDrivenDMD
using LinearAlgebra
using Test
using StatsBase
using StableRNGs
using OrdinaryDiffEq

rng = StableRNG(42)

@testset "Linear Forced System" begin
    function linear(du, u, p, t)
        du[1] = -0.9 * u[1] + 0.1 * u[2]
        du[2] = -0.8 * u[2] + 3.0sin(t)
    end

    u0 = [1.0; 1.0]
    prob_cont = ODEProblem(linear, u0, (0.0, 30.0))
    sol_cont = solve(prob_cont, Tsit5())
    U = reshape(map(t -> sin(t), sol_cont.t), 1, length(sol_cont))

    ddprob = DataDrivenProblem(sol_cont, U = U, use_interpolation = true)

    for alg in [DMDPINV(); DMDSVD(); TOTALDMD(3, DMDPINV())]
        res = solve(ddprob, alg, options = DataDrivenCommonOptions(digits = 1))
        koopman_res = first(get_results(res))
        @test r2(res) >= 0.95
        @test rss(res) <= 2.0
        @test loglikelihood(res) >= 160.0
        @test get_parameter_values(res.basis) ≈ [-0.8, -0.7, 2.9]
        @test get_inputmap(koopman_res) ≈ [0.0; 3.0;;]
        @test Matrix(get_operator(koopman_res)) ≈ [-0.9 0.1; 0.0 -0.8]
    end
end

@testset "Linear Forced Unstable System" begin
    # Define measurements from unstable system with known control input
    X = [4 2 1 0.5 0.25; 7 0.7 0.07 0.007 0.0007]
    U = [-4 -2 -1 -0.5 0] # Add the zero because we probably know the input here
    B = [1; 0;;]

    ddprob = DiscreteDataDrivenProblem(X, t = 1:5, U = U)

    for alg in [DMDPINV(); DMDSVD(); TOTALDMD(2, DMDPINV())]
        res = solve(ddprob, alg, control_input = B)
        koopman_result = first(get_results(res))
        @test r2(res)≈0.95 atol=5e-1
        @test dof(res) == 3
        @test rss(res) <= eps()
        @test Matrix(get_operator(koopman_result)) ≈ [1.5 0; 0 0.1]
        @test get_inputmap(koopman_result) == B
        @test get_outputmap(koopman_result) ≈ I(2)
    end
end
