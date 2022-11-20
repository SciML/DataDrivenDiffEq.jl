using DataDrivenDiffEq
using DataDrivenLux
using Random
using Lux
using Test
using Distributions
using DataDrivenDiffEq.StatsBase

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

        d = DecisionNode(2, 1, sin, simplex = GumbelSoftmax())
        ps, st = Lux.setup(rng, d)
        st_x = update_state(d, ps, st)
        st_X = update_state(d, ps, st)
        y, st_x = d(x, ps, st_x)
        Y, st_X = d(X, ps, st_X)
        @test y == sin(x[st_x.input_id])
        @test Y == permutedims(map(xi -> sin(xi[st_X.input_id]), eachcol(X)))
    end

    @testset "Skip" begin
        rng = Random.default_rng()
        x = randn(rng, 2)
        X = randn(rng, 2, 10)
        d = DecisionNode(2, 1, sin, simplex = Softmax(), skip = true)
        ps, st = Lux.setup(rng, d)
        st_x = update_state(d, ps, st)
        st_X = update_state(d, ps, st)
        y, st_x = d(x, ps, st_x)
        Y, st_X = d(X, ps, st_X)
        @test y == vcat(sin(x[st_x.input_id]), x)
        @test Y == vcat(permutedims(map(xi -> sin(xi[st_X.input_id]), eachcol(X))), X)
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

@testset "DecisionLayer" begin
    @testset "No Skip" begin
        rng = Random.default_rng()
        d = DecisionLayer(3, (1, 1), (sin, exp), simplex = Softmax())
        ps, st = Lux.setup(rng, d)
        st_x = update_state(d, ps, st)
        st_X = update_state(d, ps, st)
        x = randn(rng, 3)
        X = randn(rng, 3, 10)
        y, st_x = d(x, ps, st_x)
        Y, st_X = d(X, ps, st_X)
        id_x = reduce(vcat, map(x -> x.input_id, values(st_x)))
        id_X = reduce(vcat, map(x -> x.input_id, values(st_X)))
        @test y == map(zip((sin, exp), x[id_x])) do (fi, xi)
            fi(xi)
        end
        @test Y == reduce(vcat, map(zip((sin, exp), eachrow(X[id_X, :]))) do (fi, xi)
                              permutedims(fi.(xi))
                          end)
        @test sum(x -> sum(x.loglikelihood), values(st_x)) == logpdf(d, ps, st_x)
        @test prod(x -> prod(exp, x.loglikelihood), values(st_x)) == pdf(d, ps, st_x)
    end
    @testset "Skip" begin
        rng = Random.default_rng()
        d = DecisionLayer(3, (1, 1), (sin, exp), skip = true, simplex = Softmax())
        ps, st = Lux.setup(rng, d)
        st_x = update_state(d, ps, st)
        st_X = update_state(d, ps, st)
        x = randn(rng, 3)
        X = randn(rng, 3, 10)
        y, _ = d(x, ps, st_x)
        Y, _ = d(X, ps, st_X)
        id_x = reduce(vcat, map(x -> x.input_id, values(st_x)))
        id_X = reduce(vcat, map(x -> x.input_id, values(st_X)))
        @test y == vcat(map(zip((sin, exp), x[id_x])) do (fi, xi)
                            fi(xi)
                        end, x)
        @test Y == vcat(reduce(vcat, map(zip((sin, exp), eachrow(X[id_X, :]))) do (fi, xi)
                              permutedims(fi.(xi))
                          end), X)
        @test sum(x -> sum(x.loglikelihood), values(st_x)) == logpdf(d, ps, st_x)
        @test prod(x -> prod(exp, x.loglikelihood), values(st_x)) == pdf(d, ps, st_x)
    end
end

@testset "Layered DAG" begin
    @testset "No Skip" begin
        rng = Random.default_rng()
        chain = DataDrivenLux.LayeredDAG(1, 1, 1, (1,), (sin,), simplex = Softmax())
        ps, st = Lux.setup(rng, chain)
        x = randn(rng, 1)
        st_x = update_state(chain, ps, st)
        y, st_xx = chain(x, ps, st_x)
        @test st_xx == st_x
        @test y == sin.(x)
        @test logpdf(chain, ps, st) == 0.0f0
        @test pdf(chain, ps, st) == 1.0f0

        X = randn(rng, 1, 10)
        st_X = update_state(chain, ps, st)
        Y, st_XX = chain(X, ps, st_X)
        @test st_XX == st_X
        @test Y == sin.(X)
        @test logpdf(chain, ps, st) == 0.0f0
        @test pdf(chain, ps, st) == 1.0f0
    end

    @testset "Skip" begin
        rng = Random.default_rng()
        chain = DataDrivenLux.LayeredDAG(1, 2, 1, (1,), (sin,), simplex = Softmax(),
                                         skip = true)
        ps, st = Lux.setup(rng, chain)
        x = randn(rng, 1)
        st_x = update_state(chain, ps, st)
        y, st_xx = chain(x, ps, st_x)
        @test st_xx == st_x
        @test logpdf(chain, ps, st_x) == log(0.25f0)
        @test pdf(chain, ps, st_x) == 0.25f0

        X = randn(rng, 1, 10)
        st_X = update_state(chain, ps, st)
        Y, st_XX = chain(X, ps, st_X)
        @test size(Y) == (2, 10)
    end
end

