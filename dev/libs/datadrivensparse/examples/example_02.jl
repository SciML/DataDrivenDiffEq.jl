using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenSparse
using StableRNGs

rng = StableRNG(1337)

function pendulum(u, p, t)
    x = u[2]
    y = -9.81sin(u[1]) - 0.3u[2]^3 - 3.0 * cos(u[1]) - 10.0 * exp(-((t - 5.0) / 5.0)^2)
    return [x; y]
end

u0 = [0.99π; -1.0]
tspan = (0.0, 15.0)
prob = ODEProblem(pendulum, u0, tspan)
sol = solve(prob, Tsit5(), saveat = 0.01);

X = sol[:, :] + 0.2 .* randn(rng, size(sol));
ts = sol.t;

prob = ContinuousDataDrivenProblem(X, ts, GaussianKernel(),
                                   U = (u, p, t) -> [exp(-((t - 5.0) / 5.0)^2)],
                                   p = ones(2))

@variables u[1:2] c[1:1]
@parameters w[1:2]
u = collect(u)
c = collect(c)
w = collect(w)

h = Num[sin.(w[1] .* u[1]); cos.(w[2] .* u[1]); polynomial_basis(u, 5); c]

basis = Basis(h, u, parameters = w, controls = c);
println(basis) # hide

sampler = DataProcessing(split = 0.8, shuffle = true, batchsize = 30, rng = rng)
λs = exp10.(-10:0.1:0)
opt = STLSQ(λs)
res = solve(prob, basis, opt,
            options = DataDrivenCommonOptions(data_processing = sampler, digits = 1))

system = get_basis(res)
params = get_parameter_map(system)
println(system) # hide
println(params) # hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
