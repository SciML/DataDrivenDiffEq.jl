using DataDrivenDiffEq
using DataDrivenDMD
using LinearAlgebra
using Test
using StatsBase
using StableRNGs
using OrdinaryDiffEq

rng = StableRNG(42)

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
  
    prob = DataDrivenProblem(solution, U = ufun, use_interpolation = true)
  
    @variables u[1:2] y[1:1] t
    Ψ = Basis([u; u[1]^2; y], u, controls = y, iv = t)
  
    for alg in [DMDPINV(); DMDSVD(); TOTALDMD()]
        res = solve(prob, Ψ, alg, options = DataDrivenCommonOptions(digits = 1))
        koopman_res = first(get_results(res))
        @test r2(res) >= 0.98
        @test dof(res) == 4
        @test get_inputmap(koopman_res)  ≈ [0; 1.0; 0.0;;] atol=1e-1
        @test get_outputmap(koopman_res) ≈ [1.0 0 0; 0 1 0]
        @test get_parameter_values(res.basis) ≈ [-2.0; -0.4; 0.5; 0.9]
    end
  end
  
  @testset "Slow Manifold Discrete System" begin
    function slow_manifold(du, u, p, t)
      du[1] = p[1]*u[1]
      du[2] = p[2]*u[2]+(p[1] - p[2])*u[1]^2 + p[3]*exp(-(t-10.0)/10.0)
    end
  
    u0 = [3.0; -2.0]
    tspan = (0.0, 15.0)
    p = [-0.8; -0.7; 1.0]
  
    problem = DiscreteProblem(slow_manifold, u0, tspan, p)
    solution = solve(problem, FunctionMap())
  
    ufun(u,p,t) = exp(-(t-10.0)/10.0)
  
    prob = DataDrivenProblem(solution, U = ufun)
  
    @variables u[1:2] y[1:1] t
  
    Ψ = Basis([u[1]; u[1]^2; u[2]-u[1]^2; y], u, controls = y, iv = t)
  
    for alg in [DMDPINV(); DMDSVD(); TOTALDMD()]
        res = solve(prob, Ψ, alg, options = DataDrivenCommonOptions(digits = 1))
        koopman_res = first(get_results(res))
        @info r2(res) >= 0.95
        @test dof(res) == 4
        @test get_inputmap(koopman_res)  ≈ [0; 0.0; 1.0;;] atol=1e-1
        @test get_outputmap(koopman_res) ≈ [1.0 0 0; 0 1 1]
    end
  end