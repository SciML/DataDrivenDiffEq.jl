using Revise
using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using OrdinaryDiffEq
using Plots 
using Statistics
using StatsBase
using Random

@variables u[1:2]
basis = Basis([polynomial_basis(u, 5); sin.(u); cos.(u)], u)

function pendulum(u, p, t)
    x = u[2]
    y = -9.81sin(u[1]) - 0.1u[2]^3 - 0.2 * cos(u[1])
    return [x;y]
end

u0 = [0.99π; -1.0]
tspan = (0.0, 20.0)
dt = 0.1
prob = ODEProblem(pendulum, u0, tspan)
sol = solve(prob, Tsit5(), saveat=dt)




X = Array(sol)
t = sol.t
Random.seed!(1234)
X = X .+ 1e-1*randn(size(X))

dd_prob_noisy = ContinuousDataDrivenProblem(
    X, t, GaussianKernel()
    )

plot(dd_prob_noisy)

sampler = DataSampler(Split(ratio = 1.0), Batcher(n = 3, shuffle = true, repeated = true, batchsize_min = 40))
opt = ADMM(1e-3:1e-3:5e-1)
sol = solve(dd_prob_noisy, basis, opt, maxiter = 1000, by = :min, denoise = true, normalize = true, sampler = sampler)

