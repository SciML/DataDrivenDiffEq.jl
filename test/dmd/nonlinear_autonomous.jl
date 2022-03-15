@testset "Slow Manifold System" begin
  function slow_manifold(du, u, p, t)
    du[1] = p[1]*u[1]
    du[2] = p[2]*(u[2]-u[1]^2)
  end

  u0 = [3.0; -2.0]
  tspan = (0.0, 5.0)
  p = [-0.8; -0.7]

  problem = ODEProblem(slow_manifold, u0, tspan, p)
  solution = solve(problem, Tsit5(), saveat = 0.01)


  prob = ContinuousDataDrivenProblem(solution)

  @variables u[1:2]
  Ψ = Basis([u; u[1]^2], u)
  for alg in [DMDPINV(); DMDSVD(); TOTALDMD(); FBDMD()]
    res = solve(prob, Ψ, alg, digits = 1)
    b = result(res)
    m = metrics(res)
    @test isapprox(eigvals(b), [2*p[1]; p[1]; p[2]], atol = 1e-1)
    @test all(m[:L₂] .< 1e-10)

    @test is_stable(b)
    
    @test_throws AssertionError operator(b)
    @test_nowarn generator(b)
    vls, vcs = eigen(b)
    @test vcs == modes(b)
    @test vls == frequencies(b)
    @test isapprox(outputmap(b) , [1.0 0.0 0.0; 0.0 1.0 0.0])
    @test updatable(b)

    _prob = ODEProblem((args...)->b(args...), u0, tspan, parameters(res))
    _sol = solve(_prob, Tsit5(), saveat = solution.t)
    @test norm(solution - _sol)/size(solution, 2) < 1e-1
  end
end

@testset "Slow Manifold Discrete System" begin
  function slow_manifold(du, u, p, t)
    du[1] = p[1]*u[1]
    du[2] = p[2]*u[2]+(p[1]^2 - p[2])*u[1]^2
  end

  u0 = [3.0; -2.0]
  tspan = (0.0, 10.0)
  p = [-0.8; -0.7]

  problem = DiscreteProblem(slow_manifold, u0, tspan, p)
  solution = solve(problem, FunctionMap())

  ddprob = DiscreteDataDrivenProblem(solution)

  @variables u[1:2]

  Ψ = Basis([u[1]; u[1]^2; u[2]-u[1]^2], u)

  for alg in [DMDPINV(); DMDSVD(); TOTALDMD(); FBDMD()]
    res = solve(ddprob, Ψ, alg, digits = 1)
    b = result(res)
    m = metrics(res)
    @test isapprox(eigvals(b), [p[1]; p[2]; p[1]^2], atol = 1e-1)
    @test all(m[:L₂] .< 3e-1)

    @test_throws AssertionError generator(b)
    @test_nowarn operator(b)
    vls, vcs = eigen(b)
    @test vcs == eigvecs(b)
    @test vls == eigvals(b)
    @test isapprox(outputmap(b) , [1.0 0.0 0.0; 0.0 1.0 1.0])
    @test updatable(b)
  end
end

