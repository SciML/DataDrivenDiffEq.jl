using DataDrivenDiffEq
using DataDrivenSR
using Test
using StableRNGs

rng = StableRNG(42)
X = rand(rng, 2, 50)

@testset "Simple" begin
    alg = DataDrivenSR.EQSearch(
        eq_options = Options(
            unary_operators = [sin, exp],
            binary_operators = [*], maxdepth = 1,
            seed = 42,
            verbosity = -1, progress = false
        )
    )
    f(x) = [sin(x[1]); exp(x[2])]
    Y = hcat(map(f, eachcol(X))...)
    prob = DirectDataDrivenProblem(X, Y)
    res = solve(prob, alg)
    @test r2(res) >= 0.95
    @test rss(res) <= 1.0e-5
end

@testset "Univariate" begin
    alg = DataDrivenSR.EQSearch(
        eq_options = Options(
            unary_operators = [sin, exp],
            binary_operators = [*], maxdepth = 1,
            seed = 42,
            verbosity = -1, progress = false
        )
    )

    Y = sin.(X[1:1, :])
    prob = DirectDataDrivenProblem(X, Y)
    res = solve(prob, alg)
    @test r2(res) >= 0.95
    @test rss(res) <= 1.0e-5
end

@testset "Lifted" begin
    alg = DataDrivenSR.EQSearch(
        eq_options = Options(
            unary_operators = [sin, exp],
            binary_operators = [+], maxdepth = 1,
            seed = 42,
            verbosity = -1, progress = false
        )
    )

    f(x) = [sin(x[1] .^ 2); exp(x[2] * x[1])]
    Y = hcat(map(f, eachcol(X))...)

    @variables x y
    basis = Basis([x; y; x^2; y^2; x * y], [x; y])
    prob = DirectDataDrivenProblem(X, Y)
    res = solve(prob, basis, alg)
    @test r2(res) >= 0.95
    @test rss(res) <= 1.0e-5
end
