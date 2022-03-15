@testset "Linear Forced System" begin
    function linear(du, u, p, t)
        du[1] = -0.9*u[1] + 0.1*u[2]
        du[2] = -0.8*u[2] + 3.0sin(t)
    end

    u0 = [1.0; 1.0]
    prob_cont = ODEProblem(linear, u0, (0.0, 30.0))
    sol_cont = solve(prob_cont, Tsit5())
    U = reshape(map(t->sin(t), sol_cont.t),1, length(sol_cont))

    ddprob = ContinuousDataDrivenProblem(sol_cont, U = U)

    for alg in [DMDPINV(); DMDSVD(); TOTALDMD(3, DMDPINV()); FBDMD()]
        k = solve(ddprob, alg, digits = 2)
        b = result(k)
        m = metrics(k)
        @test all(m[:L₂] .< 1e-10)
        @test Matrix(b) ≈ [-0.9 0.1; 0.0 -0.8]
        @test is_stable(b)
        @test length(controls(b)) == 1
    end
end

@testset "Linear Forced Unstable System" begin
    # Define measurements from unstable system with known control input
    X = [4 2 1 0.5 0.25; 7 0.7 0.07 0.007 0.0007]
    U = [-4 -2 -1 -0.5 0] # Add the zero because we probably know the input here
    B = Float32[1; 0]

    ddprob = DiscreteDataDrivenProblem(X, t = 1:5, U = U)

    for alg in [DMDPINV(); DMDSVD(); TOTALDMD(2, DMDPINV()); FBDMD()]
        res = solve(ddprob, alg, B = B)
        b = result(res)
        m = metrics(res)
        @test Matrix(b) ≈[1.5 0; 0 0.1]
        @test !is_stable(b)
        @test all(m[:L₂] .< 1e-10)
        @test length(controls(b)) == 1
    end
end

