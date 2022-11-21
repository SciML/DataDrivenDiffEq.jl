using Revise
using DataDrivenDiffEq
using DataDrivenLux
using Lux
using LinearAlgebra
using Random

X = randn(2, 10)
Y = permutedims([sin.(X[1, :] .+ X[2, :]);; exp.(X[2, :])])
Y .+= 0.05 * randn(size(Y))
alg = RandomSearch(populationsize = 2, functions = (exp, sin, +, *), arities = (1, 1, 2, 2))

# Overall with 2 candidates and @btime
# 2.648 s (6564691 allocations: 402.98 MiB)
function traincache(alg, data::Tuple, iterations = 1)
    # For 100 candidates with @time
    # 3.311180 seconds (18.08 M allocations: 2.176 GiB, 10.09% gc time, 57.74% compilation time)
    # For 1 with @btime ( we need 2 since we want to update one)
    # 2.074 s (5216575 allocations: 286.07 MiB
    # For 2 
    # 2.087 s (5313192 allocations: 300.11 MiB)
    # Datasize = 2, 10
    # 2.196 s (5158630 allocations: 275.45 MiB)
    cache = SearchCache(alg, data...)
    for i in 1:iterations
        # Datasize = 2, 100
        # For 100 candidates with @time
        #  1.308970 seconds (9.95 M allocations: 1.394 GiB, 12.93% gc time, 35.43% compilation time)
        # For 2 with @btime
        # 696.167 μs (8829 allocations: 1.17 MiB)
        # Datasize = 2, 10
        # 85.291 μs (1449 allocations: 93.02 KiB)
        DataDrivenLux.update!(cache, data...)
    end
    return cache
end

data = (X, Y)
alg = RandomSearch(populationsize = 20, functions = (exp, sin, +, *, /),
                   arities = (1, 1, 2, 2, 2), rng = Random.seed!(42))

cache = traincache(alg, data, 100)
cache
@btime SearchCache($alg, $X, $Y)
cache = SearchCache(alg, data...)
@btime DataDrivenLux.update!($cache, $X, $Y)
@btime traincache($alg, $data)

cache = traincache(alg, data)

@variables x[1:2]
x = collect(x)
Ŷ, _ = cache.model(x, cache.p, cache.candidates[1].st)
input_nodes = DataDrivenLux.__get_input(cache.candidates[1].st)
DataDrivenLux.__get_unique_nodes(reverse(input_nodes))
cache.candidates[1].st.layer_3
get_scales(cache.candidates[1])

pl = scatter(Y', color = :red, legend = false)
for i in 1:1:5
    Ŷ, _ = cache.model(X, cache.p, cache.candidates[i].st)
    plot!(Ŷ', alpha = 0.2, color = :black, legend = false, ylim = 1.1 .* extrema(Y))
end
display(pl)
