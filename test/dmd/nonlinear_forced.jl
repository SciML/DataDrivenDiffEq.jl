@testset "Slow Manifold System" begin
  function slow_manifold(du, u, p, t)
    du[1] = p[1]*u[1]
    du[2] = p[2]*(u[2]-u[1]^2) + p[3]*sin(t^2)
  end

  u0 = [3.0; -2.0]
  tspan = (0.0, 5.0)
  p = [-2.0; -0.5; 1.0]


  problem = ODEProblem(slow_manifold, u0, tspan, p)
  solution = solve(problem, Tsit5(), saveat = 0.01)
  
  ufun(u,p,t) = sin(t^2)

  prob = ContinuousDataDrivenProblem(solution, U = ufun)

  @variables u[1:2] y[1:1] t
  Ψ = Basis([u; u[1]^2; y], u, controls = y, iv = t)

  for alg in [DMDPINV(); DMDSVD(); TOTALDMD()]
    res = solve(prob, Ψ, alg, digits = 1)
    b = result(res)
    m = metrics(res)
    @test isapprox(eigvals(b), [2*p[1]; p[1]; p[2]], atol = 1e-1)
    @test m.Error/size(X, 2) < 3e-1

    # TODO This does not work right now, but it should
    #sdict = Dict([y[1] => sin(t^2)])

    #for (i,eq) in enumerate(equations(b))
    #  lhs, rhs = eq.lhs, eq.rhs
    #  b[i] = Num(lhs) ~ substitute(Num(rhs), sdict)
    #end

    #_prob = ODEProblem((args...)->b(args...), u0, tspan, parameters(res))
    #_sol = solve(_prob, Tsit5(), saveat = solution.t)
    #@test norm(solution - _sol)/size(X, 2) < 5e-1
  end
end
