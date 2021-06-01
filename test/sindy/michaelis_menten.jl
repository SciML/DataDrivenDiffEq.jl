using OrdinaryDiffEq
using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using Random
using Test


function michaelis_menten(u, p, t)
    [0.6 - 1.5u[1]/(0.3+u[1])]
end

u0 = [0.5]
tspan = (0.0, 4.0)

problem_1 = ODEProblem(michaelis_menten, u0, tspan)
solution_1 = solve(problem_1, Tsit5(), saveat = 0.1)
problem_2 = ODEProblem(michaelis_menten, 2*u0, tspan)
solution_2 = solve(problem_2, Tsit5(), saveat = 0.1)
X = [solution_1[:,:] solution_2[:,:]]
ts = [solution_1.t; solution_2.t]

DX = similar(X)
for (i, xi) in enumerate(eachcol(X))
    DX[:, i] = michaelis_menten(xi, [], ts[i])
end

@parameters t
@variables u[1:2]
h = [monomial_basis(u[1:1], 4)...]
basis = Basis([h; h .* u[2]], u)

@testset "Ideal data" begin

    prob = ContinuousDataDrivenProblem(X, ts, DX = DX)

    opts = [ImplicitOptimizer(5e-1);ImplicitOptimizer(0.4:0.1:0.7)]
    for opt in opts
        res = solve(prob, basis, opt, normalize = false, denoise = false, maxiter = 10000)
        m = metrics(res)
        println(m)
        @test m.Error < 5e-1
        @test m.AICC < 23.0
        @test m.Sparsity == 4
    end

    for opt in [ADM(5e-1); ADM(4e-1:1e-3:5e-1)]
        res = solve(prob, basis, opt, normalize = false, denoise = false, maxiter = 1000)
        m = metrics(res)
        @test m.Error < 8e-1
        @test m.AICC < 23.0
        @test m.Sparsity == 4
    end
end

Random.seed!(2345)
X = X .+ 1e-3*randn(size(X))

@testset "Noisy data" begin


    prob = ContinuousDataDrivenProblem(X, ts, GaussianKernel())

    for opt in [ImplicitOptimizer(4e-1);ImplicitOptimizer([0.3; 0.4; 0.5])]
        res = solve(prob, basis, opt, normalize = true, denoise = true)
        m = metrics(res)
        @test m.Error < 3e-1
        @test m.AICC < 35.0
        @test m.Sparsity == 4
    end

    # ADM does not play well with the interpolation
    prob = ContinuousDataDrivenProblem(X, ts, GaussianKernel())

    for opt in [ADM(0.01:0.01:4e-1)]
        res = solve(prob, basis, opt, normalize = false, denoise = false)
        m = metrics(res)
        @test m.Error < 5e-1
        @test m.AICC < 12.0
        @test m.Sparsity == 4
    end
end