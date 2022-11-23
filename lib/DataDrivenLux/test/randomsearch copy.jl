using Revise
using DataDrivenDiffEq
using DataDrivenLux
using Lux
using LinearAlgebra
using Random
using Distributions
using Plots

X = reduce(vcat, map(1:2) do i 
    permutedims(map(1:20) do j
        rand(truncated(Normal(), -2, 2))
    end)
end)

Y = permutedims([sin.(-1.33 .* X[1, :]);; exp.(X[2,:])])
Y .+= 0.1 * randn(size(Y))
prob = DirectDataDrivenProblem(X, Y)
plot(prob)
@variables x[1:2]
@parameters p [bounds=(-2.0,2.0), dist = truncated(Normal(0.0, 2.0), -2.0, 2.0)]

b = Basis([x;p], x, parameters = [p])

alg = RandomSearch(populationsize = 50, functions = (exp, sin, +, *,/),
                    n_layers = 3,
                   arities = (1, 1, 2, 2,2), rng = Random.seed!(12),
                   use_protected = true, 
                   loss = bic, keep = 0.2, distributed = false)

res = solve(prob, b, alg, options = DataDrivenCommonOptions(progress = true, maxiters = 2_000))
println(res)
println(get_basis(res))
println(get_parameter_map(get_basis(res)))
## Some Statistics
rss(res)
loglikelihood(res)
r2(res)
plot(plot(prob), plot(res))

##
