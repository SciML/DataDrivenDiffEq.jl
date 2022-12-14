using DataDrivenDiffEq
using DataDrivenLux
using IntervalArithmetic
using Random
using Lux
using Test

states = collect(PathState(-10.0..10.0, (0, i)) for i in 1:1)
f(x,y,z) = x*y-z
fs = (sin, +, f)
arities = (1,2,3)
x = randn(1)
X = randn(1,10)

@testset "Single Layer" begin 
    dag = LayeredDAG(1, 2, 1, arities, fs)
    @test length(dag) == 2
    rng = Random.seed!(33)
    ps, st = Lux.setup(rng, dag)
    out_state, new_st = dag(states, ps, st)
    y, _ = dag(x, ps, new_st)
    Y, _ = dag(X, ps, new_st)
    @test y ==[sin.(x[1]); sin.(x[1])]
    @test Y == [sin.(X[1:1,:]); sin.(X[1:1,:])]
    @test exp(sum(sum ∘ values, values(DataDrivenLux.get_loglikelihood(dag, ps, new_st)))) == 1f0
end

@testset "Two Layer Skip" begin
    dag = LayeredDAG(1, 2, 2, arities, fs, skip = true)
    @test length(dag) == 3
    rng = Random.seed!(33)
    ps, st = Lux.setup(rng, dag)
    out_state, new_st = dag(states, ps, st)
    y, _ = dag(x, ps, new_st)
    Y, _ = dag(X, ps, new_st)
    @test y ==[sin.(x[1]) .+ x[1]; x[1]]
    @test Y == [sin.(X[1:1,:]) .+ X[1:1,:]; X[1:1,:]]
    @test DataDrivenLux.get_loglikelihood(dag, ps, new_st, out_state) == Float32[-2.7725887, -1.3862944]
end
