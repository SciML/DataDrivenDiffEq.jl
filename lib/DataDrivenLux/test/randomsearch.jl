using Distributed
nprocs()
addprocs(2)

@everywhere begin 
    import Pkg
    Pkg.activate(joinpath(pwd(), "TestDistributedLux"))
    Pkg.instantiate()
    Pkg.precompile()
    using DataDrivenDiffEq
    using DataDrivenLux
    using Lux
    using LinearAlgebra
    using Random
    using Distributions
end

using Plots


X = reduce(vcat, map(1:2) do i 
    permutedims(map(1:10) do j
        rand(truncated(Normal(), -2, 2))
    end)
end)

Y = permutedims([sin.(-1.33 .* X[1, :]);; exp.(X[2,:])])
Y .+= 0.1 * randn(size(Y))
plot(Y')
prob = DirectDataDrivenProblem(X, Y)
@variables x[1:2]
@parameters p [bounds=(-2.0,2.0), dist = truncated(Normal(0.0, 2.0), -2.0, 2.0)]

b = Basis([x;p], x, parameters = [p])

alg = RandomSearch(populationsize = 200, functions = (exp, sin, +, *),
                   arities = (1, 1, 2, 2), rng = Random.seed!(11), 
                   loss = aicc, keep = 1, distributed = true)

res = solve(prob, b, alg, options = DataDrivenCommonOptions(progress = true, maxiters = 5_000))

println(res)
println(get_basis(res))
println(get_parameter_map(get_basis(res)))
## Some Statistics
rss(res)
loglikelihood(res)
r2(res)
plot(plot(prob), plot(res))

##


# Overall with 2 candidates and @btime
# 2.648 s (6564691 allocations: 402.98 MiB)
function traincache(alg, basis, data::Tuple, iterations = 1)
    # For 100 candidates with @time
    # 3.311180 seconds (18.08 M allocations: 2.176 GiB, 10.09% gc time, 57.74% compilation time)
    # For 1 with @btime ( we need 2 since we want to update one)
    # 2.074 s (5216575 allocations: 286.07 MiB
    # For 2 
    # 2.087 s (5313192 allocations: 300.11 MiB)
    # Datasize = 2, 10
    # 2.196 s (5158630 allocations: 275.45 MiB)
    cache = SearchCache(alg, basis, data...)
    p = Progress(iterations, 1)
    for i in 1:iterations
        # Datasize = 2, 100
        # For 100 candidates with @time
        #  1.308970 seconds (9.95 M allocations: 1.394 GiB, 12.93% gc time, 35.43% compilation time)
        # For 2 with @btime
        # 696.167 μs (8829 allocations: 1.17 MiB)
        # Datasize = 2, 10
        # 85.291 μs (1449 allocations: 93.02 KiB)
        # New
        # 33.750 μs (495 allocations: 31.81 KiB)
        DataDrivenLux.update!(cache)
        Progressmeter.update!(p, i)        
    end
    return cache
end

cache = traincache(alg, b, (X, Y), 1_000)
@info cache

best_cache = cache.candidates[1]
p_opt = DataDrivenLux.get_parameters(best_cache)

Ŷ, _ = cache.model(b(cache.dataset, p_opt), cache.p, best_cache.st)
pl = scatter(Y', color = :red, legend = false)
for i in 1:1:5
    Ŷ, _ = cache.model(b(cache.dataset, p_opt), cache.p, best_cache.st)
    plot!(Ŷ', alpha = 0.2, color = :black, legend = false, ylim = 5.0 .* extrema(Y))
end
display(pl)
p_opt
Ŷ, _ = cache.model(b(x, p_opt), cache.p, best_cache.st)
println(Ŷ)