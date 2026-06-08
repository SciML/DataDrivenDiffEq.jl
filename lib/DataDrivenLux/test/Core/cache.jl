using DataDrivenLux
using Random
using Lux
using Test
using Distributions
using DataDrivenDiffEq
using StableRNGs

rng = StableRNG(1234)
# Dummy stuff
X = randn(rng, 1, 10)
Y = sin.(X[1:1, :])
@variables x[1:1]
dummy_basis = Basis(x, x)
dummy_problem = DirectDataDrivenProblem(X, Y)

# We have 1 Choices in the first layer, 2 in the last
alg = RandomSearch(
    populationsize = 10, functions = (sin,), arities = (1,),
    rng = rng, loss = rss, keep = 1, distributed = false
)

cache = DataDrivenLux.init_cache(alg, dummy_basis, dummy_problem)
rss_wrong = sum(abs2, Y .- X)

@test 2 <= length(unique(map(rss, cache.candidates))) <= 3
@test all(x -> (rss(x) == rss_wrong) || (rss(x) == 0.0), cache.candidates)
@test sum(cache.keeps) == 1
@test all(iszero, cache.ages)

# Update the cache
DataDrivenLux.update_cache!(cache)
@test length(unique(map(rss, cache.candidates))) == 2
@test all(x -> (rss(x) == rss_wrong) || (rss(x) == 0.0), cache.candidates)
@test sum(cache.keeps) == 1
@test sum(cache.ages) == 1
@test (0.0, 1) == findmin(rss, cache.candidates)

# Update another 10 times
foreach(1:10) do i
    return DataDrivenLux.update_cache!(cache)
end

@test length(unique(map(rss, cache.candidates))) == 2
@test all(x -> (rss(x) == rss_wrong) || (rss(x) == 0.0), cache.candidates)
@test sum(cache.keeps) == 1
@test sum(cache.ages) == 11
@test (0.0, 1) == findmin(rss, cache.candidates)
@test_nowarn basis = DataDrivenLux.convert_to_basis(cache.candidates[1])
basis = DataDrivenLux.convert_to_basis(cache.candidates[1])
@test all(isequal.(map(x -> x.rhs, equations(basis)), [sin(x[1])]))
