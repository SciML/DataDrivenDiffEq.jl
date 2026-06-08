using DataDrivenDiffEq
using DataDrivenLux
using Lux
using LinearAlgebra
using Random
using Distributions
using Test
using StableRNGs
using IntervalArithmetic

rng = StableRNG(1234)
# Dummy stuff
X = randn(rng, 1, 40)
Y = sin.(X[1:1, :])
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

for (data, _interval) in zip(
        (X, Y, 1:size(X, 2)),
        (dummy_dataset.x_intervals[1], dummy_dataset.y_intervals[1], dummy_dataset.t_interval)
    )
    @test isequal_interval(_interval, interval(extrema(data)))
end

# We have 1 Choices in the first layer, 2 in the last
alg = RandomSearch(;
    populationsize = 10, functions = (sin, exp, *), arities = (1, 1, 2), rng,
    n_layers = 2, loss = rss, keep = 2
)

res = solve(
    dummy_problem, alg,
    options = DataDrivenCommonOptions(
        maxiters = 50, progress = parse(Bool, get(ENV, "CI", "false")), abstol = 0.0
    )
)
@test rss(res) <= 1.0e-2
@test aicc(res) <= -100.0
@test r2(res) >= 0.95
results = get_results(res)
@test length(results) == 1
