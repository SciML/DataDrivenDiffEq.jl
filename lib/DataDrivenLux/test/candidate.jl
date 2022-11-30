using DataDrivenLux
using Random
using Lux
using Test
using Distributions
using DataDrivenDiffEq

rng = Random.seed!(1234)

# We first define a dummies

X = randn(rng, 2, 10)
Y = randn(rng, 1, 10)
@variables x[1:2]
dummy_basis = Basis(x,x)
dummy_data = DataDrivenLux.Dataset(X, Y)
dummy_dag = DataDrivenLux.LayeredDAG(2, 1, 1, (1,), (sin,), simplex = Softmax(),
                                 skip = true)

ps, st = Lux.setup(rng, dummy_dag)

# Sample a single candidate
candidate = DataDrivenLux.Candidate(dummy_dag, ps, st, dummy_basis, dummy_data)

# Checkout function interface
ŷ = candidate(dummy_data, ps, candidate.st, candidate.parameters)
@test size(ŷ) == (1,10)
@test sum(abs2, ŷ .- Y) == rss(candidate)

# Assert the state
@test length(candidate.outgoing_path) == 1
@test 2 <= length(DataDrivenLux.get_nodes(candidate)) <= 3
rss_ = rss(candidate)
dof_ = dof(candidate)
aic_ = aic(candidate)
bic_ = bic(candidate)

# Resample

@test_nowarn DataDrivenLux.optimize_candidate!(candidate, ps, dummy_data, LBFGS(), Optim.Options())
@test_nowarn DataDrivenLux.update_values!(candidate, ps, dummy_data)
@test rss(candidate) != rss_
@test 2 <= dof(candidate) <= 3
@test aic(candidate) != aic_
@test bic(candidate) != bic_

ŷ = candidate(dummy_data, ps, candidate.st, candidate.parameters)
@test size(ŷ) == (1,10)
@test sum(abs2, ŷ .- Y) == rss(candidate)
@test size(DataDrivenLux.get_scales(candidate)) == (1,)
@test size(DataDrivenLux.get_parameters(candidate)) == (0,)
@test_nowarn DataDrivenLux.get_loglikelihood(candidate, ps)