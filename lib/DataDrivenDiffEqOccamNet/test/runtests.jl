using DataDrivenDiffEq, DataDrivenDiffEqOccamNet

@testset "OccamNet Flux API" begin
    Random.seed!(1223)
    # Generate a multivariate function for OccamNet
    X = rand(2,10)
    f(x) = [sin(π*x[2]+x[1]); exp(x[2])]
    Y = hcat(map(f, eachcol(X))...)

    net = OccamNet(2, 2, 3, Function[sin, +, *, exp], skip = true, constants = Float64[π])
    initial_loss = sum([sum(abs2, net(X)-Y) for i in 1:100]) / 100
    Flux.train!(net, X, Y, ADAM(1e-2), 1000, routes = 100, nbest = 3)
    final_loss = sum([sum(abs2, net(X)-Y) for i in 1:100]) / 100
    @test initial_loss >= final_loss

    # Set the temperature low to get the best
    @test_nowarn set_temp!(net, 0.01)
    @variables x[1:2]
    x = Symbolics.scalarize(x)
    # Get the best route
    @test_nowarn rand(net)
    route = rand(net)
    @test all(probability(net, route) .> 0.9)
    # Use simplify for ordering
    eqs = simplify.(net(x, route))
    @test isequal(eqs, simplify.(Num[sin(π*x[2]+x[1]); exp(x[2])]))
end

@testset "OccamNet Solve API" begin

    sralg = OccamSR(layers = 3)
    X = rand(2,20)
    Y = permutedims(sin.(π*X[1,:]+X[2,:]))

    ddprob = DirectDataDrivenProblem(X, Y)

    res = solve(ddprob, sralg, ADAM(1e-2), max_iter = 1000, progress = false, routes = 100, temperature = 1.0)

    basis = result(res)
    m = metrics(res)
    @test all(m[:L₂] .< eps())
    @test all(m[:AIC] .> 1000.0)
end