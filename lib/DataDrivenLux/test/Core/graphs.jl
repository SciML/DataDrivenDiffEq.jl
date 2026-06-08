using DataDrivenDiffEq
using DataDrivenLux
using IntervalArithmetic
using Random
using Lux
using Test
using ComponentArrays
using StableRNGs

states = collect(PathState(interval(-10.0, 10.0), (0, i)) for i in 1:1)
f(x, y, z) = x * y - z
fs = (sin, +, f)
arities = (1, 2, 3)
x = randn(1)
X = randn(1, 10)

@testset "Single Layer" begin
    dag = LayeredDAG(1, 2, 1, arities, fs)
    rng = StableRNG(33)
    ps, st = Lux.setup(rng, dag)
    out_state, new_st = dag(states, ps, st)
    y, _ = dag(x, ps, new_st)
    Y, _ = dag(X, ps, new_st)
    @test y == [sin.(x[1]); sin.(x[1])]
    @test Y == [sin.(X[1:1, :]); sin.(X[1:1, :])]
    @test exp(
        sum(
            sum ∘ values, values(DataDrivenLux.get_loglikelihood(dag, ps, new_st))
        )
    ) == 1.0f0
end

@testset "Two Layer Skip" begin
    dag = LayeredDAG(1, 2, 2, arities, fs, skip = true)
    rng = StableRNG(11)
    ps, st = Lux.setup(rng, dag)
    ps = ComponentVector(ps)
    out_state, new_st = dag(states, ps, st)
    y, _ = dag(x, ps, new_st)
    Y, _ = dag(X, ps, new_st)
    @test y == [sin.(x[1]) .+ x[1]; x[1]]
    @test Y == [sin.(X[1:1, :]) .+ X[1:1, :]; X[1:1, :]]
    @test DataDrivenLux.get_loglikelihood(dag, ps, new_st, out_state) ==
        sum(Float32[-2.7725887, -1.3862944])
end
