@testset "SymbolicRegression" begin
    Random.seed!(1223)
    # Generate a multivariate function for OccamNet
    X = rand(2,20)
    f(x) = [sin(x[1]); exp(x[2])]
    Y = hcat(map(f, eachcol(X))...)
    # Define the options
    opts = EQSearch([+, *, sin, exp], maxdepth = 1)
    # Define the problem
    prob = DirectDataDrivenProblem(X, Y)
    # Solve the problem
    res = solve(prob, opts, numprocs = 0, multithreading = false)
    sys = result(res)

    x = states(sys)
    m = metrics(res)

    @test m.Complexity == 4
    @test m.Error <= eps()
    @test m.AICC == Inf
    @test isequal([x.rhs for x in equations(sys)], [sin(x[1]); exp(x[2])])
end
