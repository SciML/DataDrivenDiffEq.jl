using DataDrivenDiffEq
using DataDrivenLux
using IntervalArithmetic
using Distributions
using Random
using Lux
using Test
using StableRNGs

@testset "Candidate without choice" begin 
    fs = (x->x^2,)
    arities = (1,)
    dag = LayeredDAG(1, 1, 1, arities, fs, skip = true)
    rng = Random.seed!(25)
    X = permutedims(collect(0:0.1:10.0))
    Y = X .* X .+ 0.1*randn(rng, size(X))
    @variables x
    basis = Basis([x], [x])

    dataset = Dataset(X, Y)
    rng = StableRNG(3)
    candidate = DataDrivenLux.Candidate(rng, dag, basis, dataset)
    @test nobs(candidate) == 101
    @test rss(candidate) <= 1.3
    @test r2(candidate) ≈ 1.0

    @test DataDrivenLux.get_scales(candidate) ≈ ones(Float64, 1)
    @test isempty(DataDrivenLux.get_parameters(candidate))
    @test_nowarn DataDrivenLux.optimize_candidate!(candidate, dataset; options = Optim.Options())
end

@testset "Candidate with parametes" begin
    fs = (exp,)
    arities = (1,)
    dag = LayeredDAG(1, 1, 1, arities, fs, skip = true)
    X = permutedims(collect(0:0.1:3.0))
    Y = sin.(2.0*X) 
    @variables x
    @parameters p [bounds = (1.0, 2.5), dist=Normal(1.75,1.0)]
    basis = Basis([sin(p*x)], [x], parameters = [p])

    dataset = Dataset(X, Y)
    rng = StableRNG(2)
    candidate = DataDrivenLux.Candidate(rng, dag, basis, dataset)
    candidate.outgoing_path
    DataDrivenLux.optimize_candidate!(candidate, dataset)
    DataDrivenLux.get_parameters(candidate)
    @test DataDrivenLux.get_scales(candidate) ≈ [1e-5]
    @test rss(candidate) <= 1e-10
    @test r2(candidate) ≈ 1.0
    @test DataDrivenLux.get_parameters(candidate) ≈ [2.0] atol=1e-2
end
