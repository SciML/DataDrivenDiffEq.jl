@testset "Slow Manifold System" begin
    function slow_manifold(du, u, p, t)
        du[1] = p[1] * u[1]
        du[2] = p[2] * (u[2] - u[1]^2) + p[3] * sin(t^2)
    end

    u0 = [3.0; -2.0]
    tspan = (0.0, 5.0)
    p = [-2.0; -0.5; 1.0]

    problem = ODEProblem(slow_manifold, u0, tspan, p)
    solution = solve(problem, Tsit5(), saveat = 0.01)

    ufun(u, p, t) = sin(t^2)

    prob = ContinuousDataDrivenProblem(solution, U = ufun)

    @variables u[1:2] y[1:1] t
    Ψ = Basis([u; u[1]^2; y], u, controls = y, iv = t)

    for alg in [DMDPINV(); DMDSVD(); TOTALDMD(); FBDMD()]
        res = solve(prob, Ψ, alg, digits = 1)
        b = result(res)
        m = metrics(res)
        @test isapprox(eigvals(b), [2 * p[1]; p[1]; p[2]], atol = 1e-1)
        @test all(m[:L₂] .< 3e-1)
    end
end

@testset "Slow Manifold Discrete System" begin
    function slow_manifold(du, u, p, t)
        du[1] = p[1] * u[1]
        du[2] = p[2] * u[2] + (p[1] - p[2]) * u[1]^2 + p[3] * exp(-(t - 10.0) / 10.0)
    end

    u0 = [3.0; -2.0]
    tspan = (0.0, 10.0)
    p = [-0.8; -0.7; 1.0]

    problem = DiscreteProblem(slow_manifold, u0, tspan, p)
    solution = solve(problem, FunctionMap())

    ufun(u, p, t) = exp(-(t - 10.0) / 10.0)

    ddprob = DiscreteDataDrivenProblem(solution, U = ufun)

    @variables u[1:2] y[1:1] t

    Ψ = Basis([u[1]; u[1]^2; u[2] - u[1]^2; y], u, controls = y, iv = t)

    for alg in [DMDPINV(); DMDSVD(); TOTALDMD(); FBDMD()]
        res = solve(ddprob, Ψ, alg, digits = 1)
        b = result(res)
        m = metrics(res)
        @test isapprox(eigvals(b), [p[1]; p[2]; p[1]^2], atol = 1e-1)
        @test all(m[:L₂] .< 3e-1)
    end
end
