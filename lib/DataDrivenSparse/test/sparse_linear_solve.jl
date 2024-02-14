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
    X = permutedims(reduce(hcat,
        (sin.(0.1 .* t), cos.(0.5 .* t), sin.(2.0 .* t .^ 2),
            cos.(0.5 .* t .^ 2), exp.(-t))))
    A = [0.68 0.0 0.0 0.0 -1.2]
    Ỹ = A * X
    Y = Ỹ + 0.01 * randn(rng, size(Ỹ))
    λ = extrema(abs.(A)[abs.(A) .> 0.0])
    true_dof = 2
    for alg in [STLSQ, ADMM, SR3]
        alg_ = alg(LinRange(0.5 * first(λ), 1.5 * last(λ), 20))
        solver = SparseLinearSolver(alg_,
            options = DataDrivenCommonOptions(verbose = false,
                maxiters = 10_000))
        res, _... = solver(X, Y)
        res = first(res)
        @test rss(res) <= 1.2
        @test aicc(res) <= -400.0
        @test true_dof == dof(res)
        @test r2(res)≈1.0 atol=6e-2
    end
end

@testset "Skinny" begin
    rng = StableRNG(52)
    # Generate data
    t = 0.0:0.5:2.0
    X = permutedims(reduce(hcat,
        (sin.(0.5 .* t), cos.(0.5 .* t), sin.(2.0 .* t .^ 2),
            cos.(0.5 .* t .^ 2), exp.(-t), randn(rng, length(t)))))
    A = [0.68 0.0 0.0 0.0 -1.2 0.0]
    Y = A * X
    λ = extrema(abs.(A)[abs.(A) .> 0.0])
    true_dof = 2
    for alg in [STLSQ, ADMM, SR3]
        alg_ = alg(LinRange(0.1, 1.6, 15))
        solver = SparseLinearSolver(alg_,
            options = DataDrivenCommonOptions(verbose = false,
                maxiters = 10_000))
        res, _... = solver(X, Y)
        res = first(res)
        @test rss(res) <= 1.5e-1
        @test aicc(res) <= -5.0
        @test true_dof == dof(res)
        @test r2(res)≈1.0 atol=1e-1
    end
end

@testset "Implicit Optimizer" begin
    t = 0.0:0.1:10.0
    X = permutedims(reduce(hcat,
        (sin.(0.5 .* t .+ 0.1), cos.(0.5 .* t), sin.(2.0 .* t .^ 2),
            cos.(0.5 .* t .^ 2 .- 0.1), exp.(-t))))
    Y = permutedims(0.5 * X[1, :] + 0.22 * X[4, :] - 2.0 * X[3, :])
    X = vcat(X, Y)
    for alg in [STLSQ, ADMM, SR3]
        opt = ImplicitOptimizer(alg())
        rescoeff, _... = opt(X, Y, options = DataDrivenCommonOptions(maxiters = 2000))
        @test vec(rescoeff)≈[0.25; 0.0; -1.0; 0.11; 0.0; -0.5] atol=5e-2
    end
end
