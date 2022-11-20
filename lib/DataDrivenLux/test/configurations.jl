using DataDrivenDiffEq
using DataDrivenLux
using Random
using Lux
using Test
using Distributions
using DataDrivenDiffEq.StatsBase

rng = Random.default_rng()
chain = LayeredDAG(1, 2, 1, (1,), (sin,), simplex = Softmax(), skip = false)

ps, st = Lux.setup(rng, chain)
st_ = update_state(chain, ps, st)
x = randn(rng, 2, 100)
y = sin.(x)

foreach(axes(y, 2)) do i
    y[2, i] = rand(Normal(y[2, i], 1.0))
end

configuration = ConfigurationCache(chain, ps, st, x, y)

@test dof(configuration) == 2
@test_nowarn loglikelihood(configuration)
@test_nowarn aic(configuration)
@test_nowarn aicc(configuration)
@test_nowarn bic(configuration)
@test_nowarn rss(configuration)
@test_nowarn nullloglikelihood(configuration)
@test r2(configuration, :CoxSnell) < 0.3

configuration = DataDrivenLux.optimize_configuration!(configuration, chain, ps, x, y)

@test dof(configuration) == 2
@test r2(configuration, :CoxSnell) ≈ 1.0
@test DataDrivenLux.get_scales(configuration)≈[0.0; 1.3] atol=5e-1
@test aic(configuration) <= -1500.0
@test iszero(DataDrivenLux.get_configuration_loglikelihood(configuration))
