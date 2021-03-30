
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
    # Build a linear basis in the output
    opt = ImplicitOptimizer(2e-1)
    res = solve(prob, basis, opt, normalize = false, denoise = false)
    m = metrics(res)
    @test m.Error < 1e-1
    @test m.AICC < 23.0
    @test m.Sparsity == 4

    opt = ImplicitOptimizer(1e-3:1e-3:5e-1)
    res = solve(prob, basis, opt, normalize = false, denoise = false)
    m = metrics(res)
    @test m.Error < 1e-1
    @test m.AICC < 23.0
    @test m.Sparsity == 4
end

X = X .+ 1e-3*randn(size(X))

@testset "Noisy data" begin


    prob = ContinuousDataDrivenProblem(X, ts, InterpolationMethod())
    # Build a linear basis in the output
    opt = ImplicitOptimizer(2e-1)
    res = solve(prob, basis, opt, normalize = false, denoise = false)
    m = metrics(res)
    @test m.Error < 1e-1
    @test m.AICC < 35.0
    @test m.Sparsity == 4

    opt = ImplicitOptimizer(1e-3:1e-3:5e-1)
    res = solve(prob, basis, opt, normalize = false, denoise = false)
    m = metrics(res)
    @test m.Error < 1e-1
    @test m.AICC < 35.0
    @test m.Sparsity == 4
end
