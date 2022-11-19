using DataDrivenDiffEq
using DataDrivenLux
using Random
using Lux
using Test
using Distributions

@testset "Decision Node" begin
    @testset "Unary" begin
        rng = Random.default_rng()
        x = randn(rng, 2)
        X = randn(rng, 2, 10)
        d = DecisionNode(2, 1, sin, simplex = Softmax())
        ps, st = Lux.setup(rng, d)
        st_x = update_state(d, ps, st)
        st_X = update_state(d, ps, st)
        y, st_x = d(x, ps, st_x)
        Y, st_X = d(X, ps, st_X)
        @test y == sin(x[st_x.input_id])
        @test Y == permutedims(map(xi -> sin(xi[st_X.input_id]), eachcol(X)))
        @test pdf(d, ps, st_x) == 0.5f0
        @test pdf(d, ps, st_X) == 0.5f0
        @test logpdf(d, ps, st_x) == log.(0.5f0)
        @test logpdf(d, ps, st_X) == log.(0.5f0)

        d = DecisionNode(2, 1, sin, simplex = GumbelSoftmax(rng))
        ps, st = Lux.setup(rng, d)
        st_x = update_state(d, ps, st)
        st_X = update_state(d, ps, st)
        y, st_x = d(x, ps, st_x)
        Y, st_X = d(X, ps, st_X)
        @test y == sin(x[st_x.input_id])
        @test Y == permutedims(map(xi -> sin(xi[st_X.input_id]), eachcol(X)))
    end
    @testset "Binary" begin
        rng = Random.default_rng()
        x = randn(rng, 2)
        X = randn(rng, 2, 10)
        d = DecisionNode(2, 2, *, simplex = Softmax())
        ps, st = Lux.setup(rng, d)
        st_x = update_state(d, ps, st)
        st_X = update_state(d, ps, st)
        y, st_x = d(x, ps, st_x)
        Y, st_X = d(X, ps, st_X)
        @test y == *(x[st_x.input_id]...)
        @test Y == permutedims(map(xi -> *(xi[st_X.input_id]...), eachcol(X)))
        @test pdf(d, ps, st_x) == 0.25f0
        @test pdf(d, ps, st_X) == 0.25f0
        @test logpdf(d, ps, st_x) == log.(0.25f0)
        @test logpdf(d, ps, st_X) == log.(0.25f0)
    end

    @testset "Ternary" begin
        rng = Random.default_rng()
        x = randn(rng, 5)
        X = randn(rng, 5, 10)
        d = DecisionNode(5, 3, *, simplex = Softmax())
        ps, st = Lux.setup(rng, d)
        st_x = update_state(d, ps, st)
        st_X = update_state(d, ps, st)
        y, st_x = d(x, ps, st_x)
        Y, st_X = d(X, ps, st_X)
        @test y == *(x[st_x.input_id]...)
        @test Y == permutedims(map(xi -> *(xi[st_X.input_id]...), eachcol(X)))
        @test pdf(d, ps, st_x) ≈ (1 / 5)^3
        @test pdf(d, ps, st_X) ≈ (1 / 5)^3
        @test logpdf(d, ps, st_x) ≈ -log.(5.0^3)
        @test logpdf(d, ps, st_X) ≈ -log.(5.0^3)
    end

    @testset "General Function" begin
        rng = Random.default_rng()
        x = randn(rng, 5)
        X = randn(rng, 5, 10)
        f(x, y, z) = 2 * (x - y) + z
        d = DecisionNode(5, 3, f, simplex = Softmax())
        ps, st = Lux.setup(rng, d)
        st_x = update_state(d, ps, st)
        st_X = update_state(d, ps, st)
        y, st_x = d(x, ps, st_x)
        Y, st_X = d(X, ps, st_X)
        @test y == f(x[st_x.input_id]...)
        @test Y == permutedims(map(xi -> f(xi[st_X.input_id]...), eachcol(X)))
        @test pdf(d, ps, st_x) ≈ (1 / 5)^3
        @test pdf(d, ps, st_X) ≈ (1 / 5)^3
        @test logpdf(d, ps, st_x) ≈ -log.(5.0^3)
        @test logpdf(d, ps, st_X) ≈ -log.(5.0^3)
    end
end
