using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using OrdinaryDiffEq

function michaelis_menten(u, p, t)
    [0.6 - 1.5u[1]/(0.3+u[1])]
end


u0 = [0.5]

problem_1 = ODEProblem(michaelis_menten, u0, (0.0, 4.0))
solution_1 = solve(problem_1, Tsit5(), saveat = 0.1)
problem_2 = ODEProblem(michaelis_menten, 2*u0, (4.0, 8.0))
solution_2 = solve(problem_2, Tsit5(), saveat = 0.1)

function michaelis_menten(X::AbstractMatrix, p, t::AbstractVector)
    reduce(hcat, map((x,ti)->michaelis_menten(x, p, ti), eachcol(X), t))
end

data = (
    Experiment_1 = (X = Array(solution_1), t = solution_1.t, DX = michaelis_menten(Array(solution_1),[], solution_1.t) ),
    Experiment_2 = (X = Array(solution_2), t = solution_2.t, DX = michaelis_menten(Array(solution_2),[], solution_2.t))
)


prob = DataDrivenDiffEq.ContinuousDataset(data)

@parameters t
D = Differential(t)
@variables u[1:1](t)
h = [monomial_basis(u[1:1], 4)...]
basis = Basis([h; h .* D(u[1])], u, implicits = D.(u), iv = t)
println(basis) # hide

sampler = DataSampler(
    Split(ratio = 0.8), Batcher(n = 10)
)

opt = ImplicitOptimizer(1e-1:1e-1:5e-1)
res = solve(prob, basis, opt,  normalize = false, denoise = false, by = :min, sampler = sampler, maxiter = 1000);
println(res) # hide

system = result(res)
println(system) # hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

