using DataDrivenDiffEq
using DataDrivenLux
using Lux
using LinearAlgebra
using Random
using Distributions
using Test
using Optimisers
using Optim
using StableRNGs

rng = StableRNG(1234)
# Dummy stuff
X = randn(rng, 2, 40)
Y = sin.(sin.(X[1:1, :]) .+ exp.(X[2:2, :]))
Y .+= 0.01 * randn(size(Y))
dummy_problem = DirectDataDrivenProblem(X, Y)
@variables x[1:1]
dummy_basis = Basis(x, x)

# Create the dataset
dummy_dataset = DataDrivenLux.Dataset(dummy_problem)

# Check the dataset
@test dummy_dataset.y == Y
@test dummy_dataset.x == X
@test isempty(dummy_dataset.u)
@test dummy_dataset.t == collect(1:size(X, 2))

@test isempty(dummy_dataset.u_intervals)

@variables x[1:2]
@parameters p [bounds = (-3.0, -1.0), dist = truncated(Normal(-2.0, 1.0), -3.0, -1.0)]

b = Basis([x; exp.(x)], x)
# We have 1 Choices in the first layer, 2 in the last
alg = Reinforce(;
    populationsize = 200, functions = (sin, exp, +), arities = (1, 1, 2), rng,
    n_layers = 3, use_protected = true, loss = bic, keep = 10, threaded = true,
    optim_options = Optim.Options(time_limit = 0.2), optimiser = AdamW(1.0e-2)
)

res = solve(
    dummy_problem, b, alg,
    options = DataDrivenCommonOptions(
        maxiters = 1000, progress = parse(Bool, get(ENV, "CI", "false")), abstol = 0.0
    )
)

@test rss(res) <= 1.0e-2
@test aicc(res) <= -100.0
@test r2(res) >= 0.95
results = get_results(res)
@test length(results) == 1
