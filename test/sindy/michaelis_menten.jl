using Test
using Random

function michaelis_menten(u, p, t)
    [0.6 - 1.5u[1]/(0.3+u[1])] # Should be 0.6*0.3+0.6u[1] - 1.5u[1] = u[2]*u[1]-0.3*u[2] 
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
u = collect(u)
h = [monomial_basis(u[1:1], 4)...]
basis = Basis([h; h .* u[2]], u)

@testset "Ideal data" begin
    
    prob = ContinuousDataDrivenProblem(X, ts, DX)
    
    opts = [ImplicitOptimizer(5e-1);ImplicitOptimizer(0.4:0.1:0.7)]
    for opt in opts
        res = solve(prob, basis, opt,u[2:2] ,normalize = false, denoise = false, maxiter = 10000)
        m = metrics(res)
        @show m[:R²]
        @test all(m[:L₂] .< 1e-1)
        @test all(m[:AIC] .> 1000.0)
        @test all(m[:R²] .> 0.6)
    end

end


Random.seed!(2345)
X = X .+ 1e-3*randn(size(X))


@testset "Noisy data" begin

    prob = ContinuousDataDrivenProblem(X, ts, GaussianKernel())

    for opt in [ImplicitOptimizer(3e-1);ImplicitOptimizer(3e-1:0.1:7e-1)]
        res = solve(prob, basis, opt, u[2:2], normalize = false, denoise = true)
        println(states(res.basis))
        m = metrics(res)
        @show m
        @test all(m[:L₂] .< 1e-1)
        @test all(m[:AIC] .> 1000.0)
        @test all(m[:R²] .> 0.9)
    end

end
