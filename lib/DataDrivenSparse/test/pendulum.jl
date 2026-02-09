using DataDrivenDiffEq
using DataDrivenSparse
using OrdinaryDiffEq
using StableRNGs
using Test
using StatsBase

# The basis definition
@variables u[1:2]
u = collect(u)
basis = Basis([polynomial_basis(u, 5); sin.(u); cos.(u)], u)

function pendulum(u, p, t)
    x = u[2]
    y = -9.81sin(u[1]) - 0.1u[2]^3 - 0.2 * cos(u[1])
    return [x; y]
end

u0 = [0.99π; -1.0]
tspan = (0.0, 20.0)
dt = 0.05
prob = ODEProblem(pendulum, u0, tspan)
sol = solve(prob, Tsit5(), saveat = dt)

@testset "Groundtruth" begin
    dd_prob = DataDrivenProblem(sol)
    for opt in [
            STLSQ(1.0e-1),
            STLSQ(1.0e-2:1.0e-2:1.0e-1, 0.0001),
            ADMM(1.0e-2),
            SR3(1.0e-2, SoftThreshold()),
            SR3(1.0e-1, ClippedAbsoluteDeviation()),
            SR3(5.0e-1),
        ]
        res = solve(
            dd_prob, basis, opt,
            options = DataDrivenCommonOptions(maxiters = 10_000, digits = 1)
        )
        @test r2(res) ≈ 0.9 atol = 1.0e-1
        @test rss(res) <= 500.0
        @test loglikelihood(res) >= 100.0
        @test 2 <= dof(res) <= 4
    end
end

@testset "Noise" begin
    X = Array(sol)
    t = sol.t

    rng = StableRNG(21)
    X_n = X .+ 1.0e-1 * randn(rng, size(X))

    dd_prob = ContinuousDataDrivenProblem(X_n, t, GaussianKernel())
    for opt in [
            STLSQ(0.5),
            STLSQ(0.5, 0.001),
            ADMM(1.0e-2),
            SR3(1.0e-2, SoftThreshold()),
            SR3(1.0e-1, ClippedAbsoluteDeviation()),
            SR3(5.0e-1),
        ]
        res = solve(
            dd_prob, basis, opt,
            options = DataDrivenCommonOptions(
                normalize = DataNormalization(ZScoreTransform),
                denoise = true,
                maxiters = 10_000, digits = 1
            )
        )
        @test r2(res) ≈ 0.9 atol = 1.0e-1
        @test rss(res) <= 100.0
        @test loglikelihood(res) >= 100.0
        @test 2 <= dof(res) <= 4
    end
end
