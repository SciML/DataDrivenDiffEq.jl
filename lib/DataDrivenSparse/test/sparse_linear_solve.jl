using DataDrivenDiffEq
using DataDrivenSparse
using Random
using LinearAlgebra
using SparseArrays
using StatsBase
using Test
using StableRNGs

@testset "Fat" begin
    rng = StableRNG(42)
    # Generate data
    t = 0.0:0.1:10.0
    X = permutedims([sin.(0.1 .* t);; cos.(0.5 .* t);; sin.(2.0 .* t .^ 2);;
                     cos.(0.5 .* t .^ 2);; exp.(-t)])
    A = sprand(rng, 1, size(X, 1), 0.4)
    Ỹ = A * X
    Y = Ỹ + 0.01 * randn(rng, size(Ỹ))
    λ = extrema(abs.(A)[abs.(A) .> 0.0])
    true_dof = sum.(eachrow(abs.(A) .> 0.0))
    for alg in [STLSQ, ADMM, SR3]
        alg_ = alg(LinRange(λ..., 20))
        solver = SparseLinearSolver(alg_,
                                    options = DataDrivenCommonOptions(verbose = false,
                                                                      maxiters = 10_000))
        res = solver(X, Y)
        res = first(res)
        @test rss(res) <= 1.2
        @test aicc(res) <= -400.0
        @test sum(abs, true_dof .- dof(res)) <= 1
        @test r2(res)≈1.0 atol=6e-2
    end
end

@testset "Skinny" begin
    rng = StableRNG(52)
    # Generate data
    t = randn(rng, 5)
    X = permutedims([sin.(0.1 .* t);; cos.(0.5 .* t);; sin.(2.0 .* t .^ 2);;
                     cos.(0.5 .* t .^ 2);; exp.(-t);; 1.0 .+ zero(t)])
    A = sprand(rng, 1, size(X, 1), 0.3)
    Ỹ = A * X
    Y = Ỹ + 0.00 * randn(rng, size(Ỹ))
    λ = extrema(abs.(A)[abs.(A) .> 0.0])
    true_dof = sum.(eachrow(abs.(A) .> 0.0))
    for alg in [STLSQ, ADMM, SR3]
        alg_ = alg(LinRange(λ..., 50))
        solver = SparseLinearSolver(alg_,
                                    options = DataDrivenCommonOptions(verbose = false,
                                                                      maxiters = 10_000))
        res = solver(X, Y)
        res = first(res)
        @test rss(res) <= 1e-2
        @test aicc(res) <= -20.0
        @test sum(abs, true_dof .- dof(res)) <= 1
        @test r2(res)≈1.0 atol=3e-2
    end
end

@testset "Implicit Optimizer" begin
    t = 0.0:0.1:10.0
    X = permutedims([sin.(0.1 .* t);; cos.(0.5 .* t);; sin.(2.0 .* t .^ 2);;
                     cos.(0.5 .* t .^ 2);; exp.(-t)])
    Y = permutedims(0.5 * X[1, :] + 0.22 * X[4, :] - 2.0 * X[3, :])
    X = vcat(X, Y)
    for alg in [STLSQ, ADMM, SR3]
        opt = ImplicitOptimizer(alg())
        rescoeff = opt(X, Y, options = DataDrivenCommonOptions())
        @test vec(rescoeff)≈[0.25; 0.0; -1.0; 0.11; 0.0; -0.5] atol=5e-2
    end
end
