using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using Plots
using LinearAlgebra
using Test
using Random

@testset "OccamNet Flux API" begin
    Random.seed!(1223)
    # Generate a multivariate function for OccamNet
    X = rand(2,10)
    f(x) = [sin(π*x[2]+x[1]); exp(x[2])]
    Y = hcat(map(f, eachcol(X))...)

    net = OccamNet(2, 2, 3, Function[sin, +, *, exp], skip = true, constants = Float64[π])
    initial_loss = sum([sum(abs2, net(X)-Y) for i in 1:100]) / 100
    @test_nowarn train!(net, X, Y, ADAM(1e-2), 1000, routes = 100, nbest = 3)
    final_loss = sum([sum(abs2, net(X)-Y) for i in 1:100]) / 100
    @test initial_loss >= final_loss

    # Set the temperature low to get the best
    @test_nowarn set_temp!(net, 0.01)
    @variables x[1:2]
    x = Symbolics.scalarize(x)
    # Get the best route
    @test_nowarn rand(net)
    route = rand(net)
    # Use simplify for ordering
    eqs = simplify.(net(x, route))
    @test isequal(eqs, simplify.(Num[sin(π*x[2]+x[1]); exp(x[2])]))
end
