using DataDrivenDiffEq
using DataDrivenLux
using IntervalArithmetic
using Random
using Lux
using Test
using StableRNGs

@testset "Layer" begin
    states = collect(PathState(interval(-10.0, 10.0), (1, i)) for i in 1:3)
    f(x, y, z) = x * y - z
    fs = (sin, +, f)
    arities = (1, 2, 3)
    x = randn(3)
    X = randn(3, 10)

    layer = FunctionLayer(3, arities, fs, id_offset = 2)
    rng = StableRNG(43)
    ps, st = Lux.setup(rng, layer)
    layer_states, new_st = layer(states, ps, st)
    @test all(
        exp.(values(DataDrivenLux.get_loglikelihood(layer, ps, new_st))) .≈
            (1 / 3, 1 / 9, 1 / 27)
    )

    intervals = map(DataDrivenLux.get_interval, layer_states)
    @test isequal_interval(intervals[1], interval(-1, 1))
    @test isequal_interval(intervals[2], interval(-20, 20))
    @test isequal_interval(intervals[3], interval(-110, 110))

    y, _ = layer(x, ps, new_st)
    Y, _ = layer(X, ps, new_st)
    @test y == [sin(x[1]); x[3] + x[1]; x[1] * x[3] - x[3]]
    @test Y == [sin.(X[1:1, :]); X[3:3, :] + X[1:1, :]; X[1:1, :] .* X[3:3, :] - X[3:3, :]]

    fs = (sin, cos, log, exp, +, -, *)
    @test DataDrivenLux.mask_inverse(log, 1, collect(fs)) == [1, 1, 1, 0, 1, 1, 1]
    @test DataDrivenLux.mask_inverse(exp, 1, collect(fs)) == [1, 1, 0, 1, 1, 1, 1]
end
