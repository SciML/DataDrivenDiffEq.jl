using DataDrivenDiffEq
using DataDrivenSparse
using Random
using LinearAlgebra
using StatsBase
using Test
using StableRNGs

@testset "Fat" begin
    rng = StableRNG(42)
    # Generate data
    t = 0.0:0.1:10.0
    X = permutedims(
        reduce(
            hcat,
            (
                sin.(0.1 .* t), cos.(0.5 .* t), sin.(2.0 .* t .^ 2),
                cos.(0.5 .* t .^ 2), exp.(-t),
            )
        )
    )
    A = [0.68 0.0 0.0 0.0 -1.2]
    Ỹ = A * X
    Y = Ỹ + 0.01 * randn(rng, size(Ỹ))
    λ = extrema(abs.(A)[abs.(A) .> 0.0])
    true_dof = 2
    for alg in [STLSQ, ADMM, SR3]
        alg_ = alg(LinRange(0.5 * first(λ), 1.5 * last(λ), 20))
        solver = SparseLinearSolver(
            alg_,
            options = DataDrivenCommonOptions(
                verbose = false,
                maxiters = 10_000
            )
        )
        res, _... = solver(X, Y)
        res = first(res)
        @test rss(res) <= 1.2
        @test aicc(res) <= -400.0
        @test true_dof == dof(res)
        @test r2(res) ≈ 1.0 atol = 6.0e-2
    end
end

@testset "Skinny" begin
    rng = StableRNG(52)
    # Generate data
    t = 0.0:0.5:2.0
    X = permutedims(
        reduce(
            hcat,
            (
                sin.(0.5 .* t), cos.(0.5 .* t), sin.(2.0 .* t .^ 2),
                cos.(0.5 .* t .^ 2), exp.(-t), randn(rng, length(t)),
            )
        )
    )
    A = [0.68 0.0 0.0 0.0 -1.2 0.0]
    Y = A * X
    λ = extrema(abs.(A)[abs.(A) .> 0.0])
    true_dof = 2
    for alg in [STLSQ, ADMM, SR3]
        alg_ = alg(LinRange(0.1, 1.6, 15))
        solver = SparseLinearSolver(
            alg_,
            options = DataDrivenCommonOptions(
                verbose = false,
                maxiters = 10_000
            )
        )
        res, _... = solver(X, Y)
        res = first(res)
        @test rss(res) <= 1.5e-1
        @test aicc(res) <= -5.0
        @test true_dof == dof(res)
        @test r2(res) ≈ 1.0 atol = 1.0e-1
    end
end

@testset "Implicit Optimizer" begin
    t = 0.0:0.1:10.0
    X = permutedims(
        reduce(
            hcat,
            (
                sin.(0.5 .* t .+ 0.1), cos.(0.5 .* t), sin.(2.0 .* t .^ 2),
                cos.(0.5 .* t .^ 2 .- 0.1), exp.(-t),
            )
        )
    )
    Y = permutedims(0.5 * X[1, :] + 0.22 * X[4, :] - 2.0 * X[3, :])
    X = vcat(X, Y)
    for alg in [STLSQ(0.1, 1.0), ADMM(), SR3()]
        opt = ImplicitOptimizer(alg)
        rescoeff, _... = opt(X, Y, options = DataDrivenCommonOptions(maxiters = 2000))
        @test vec(rescoeff) ≈ [0.25; 0.0; -1.0; 0.11; 0.0; -0.5] atol = 5.0e-2
    end
end

@testset "WyNDA online estimator" begin
    rng = StableRNG(20260708)
    t = 0.0:0.02:8.0
    X = permutedims(
        reduce(
            hcat,
            (
                ones(length(t)), sin.(0.7 .* t), cos.(1.3 .* t),
                exp.(-0.2 .* t), randn(rng, length(t)),
            )
        )
    )
    A = [0.5 -1.2 0.0 0.75 0.0; -0.25 0.0 1.5 0.0 0.1]
    Y = A * X

    coefficients, λ, iterations = WyNDA(1.0; initial_covariance = 1.0e8)(X, Y)

    @test λ == 1.0
    @test iterations == size(X, 2)
    # Finite initial covariance is equivalent to a small ridge prior in RLS.
    @test coefficients ≈ A atol = 1.0e-6

    problem = DirectDataDrivenProblem(X, Y)
    result = @test_nowarn solve(problem, WyNDA(1.0; initial_covariance = 1.0e8))
    @test result isa DataDrivenSolution
    @test result.residuals <= 1.0e-8
end

# Issue #564: Test that solve doesn't throw MethodError when coefficients are all zero
# This can happen with very small data values or high regularization
@testset "Zero coefficients handling (Issue #564)" begin
    rng = StableRNG(1111)

    # Test case 1: Very small data values that lead to zero coefficients after regularization
    N = 3
    X̂ = randn(rng, N, 50) * 1.0e-10
    Ŷ = randn(rng, 1, 50) * 1.0e-10

    @variables u[1:N]
    b = polynomial_basis(u, 2)
    basis = Basis(b, u)
    problem = DirectDataDrivenProblem(X̂, Ŷ)

    λ = 1.0e-1
    opt = ADMM(λ)
    options = DataDrivenCommonOptions()

    # This should not throw MethodError: no method matching zero(::Type{Any})
    result = @test_nowarn solve(problem, basis, opt, options = options)
    @test result isa DataDrivenSolution
    @test eltype(result.prob) == Float64

    # Test case 2: High regularization that forces all coefficients to zero
    X̂2 = randn(rng, N, 50)
    Ŷ2 = randn(rng, 1, 50)
    problem2 = DirectDataDrivenProblem(X̂2, Ŷ2)

    λ_high = 1.0e10  # Very high regularization
    opt_high = ADMM(λ_high)

    result2 = @test_nowarn solve(problem2, basis, opt_high, options = options)
    @test result2 isa DataDrivenSolution
    @test eltype(result2.prob) == Float64
end
