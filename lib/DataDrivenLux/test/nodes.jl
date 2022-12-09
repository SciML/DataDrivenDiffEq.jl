using Revise
using DataDrivenDiffEq
using DataDrivenLux
using IntervalArithmetic
using Random
using Lux
using Test


states = collect(PathState(-10.0..10.0, (1, i)) for i in 1:3)

@testset "Unary function" begin
    rng = Random.seed!(42)
    sin_node = FunctionNode(sin, 1, 3, (2,1))
    sin_node.input_mask
    ps_sin, st_sin = Lux.setup(rng, sin_node)
    sin_state, new_sin_st = sin_node(states, ps_sin, st_sin)
    @test DataDrivenLux.get_nodes(sin_state) ==((2,1),(1,2))
    @test DataDrivenLux.get_interval(sin_state) == -1..1
    @test DataDrivenLux.get_operators(sin_state) == (sin,)
    @test DataDrivenLux.get_inputs(sin_node, ps_sin, new_sin_st) == [2]
    @test DataDrivenLux.get_temperature(sin_node, ps_sin, new_sin_st) == 1f0
    @test exp(DataDrivenLux.get_loglikelihood(sin_node, ps_sin, new_sin_st)) ≈ 1/3
end

@testset "Binary function" begin
    rng = Random.seed!(233)
    add_node = FunctionNode(+, 2, 3, (2,2))
    ps_add, st_add = Lux.setup(rng, add_node)
    add_state, new_add_st = add_node(states, ps_add, st_add)
    @test DataDrivenLux.get_nodes(add_state) ==((2,2),(1,3),(1,1))
    @test DataDrivenLux.get_interval(add_state) == -20..20
    @test DataDrivenLux.get_operators(add_state) == (+,)
    @test DataDrivenLux.get_inputs(add_node, ps_add, new_add_st) == [3,1]
    @test DataDrivenLux.get_temperature(add_node, ps_add, new_add_st) == 1f0
    @test exp(DataDrivenLux.get_loglikelihood(add_node, ps_add, new_add_st)) ≈ 1/3^2
end

@testset "Ternary function" begin
    f(x,y,z) = x*y-z
    rng = Random.seed!(456)
    fnode = FunctionNode(f,3,3,(2,3))
    ps_f, st_f = Lux.setup(rng, fnode)
    f_state, new_f_st = fnode(states, ps_f, st_f)
    @test DataDrivenLux.get_interval(f_state) == -110..110
    @test DataDrivenLux.get_nodes(f_state) == ((2,3), (1,2), (1,1), (1,3))
    @test DataDrivenLux.get_operators(f_state) == (f,)
    @test DataDrivenLux.get_inputs(fnode, ps_f, new_f_st) == [2,1,3]
    @test DataDrivenLux.get_temperature(fnode, ps_f, new_f_st) == 1f0
    @test exp(DataDrivenLux.get_loglikelihood(fnode, ps_f, new_f_st)) ≈ 1/3^3
end