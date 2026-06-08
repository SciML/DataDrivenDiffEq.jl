using DataDrivenDiffEq
using LinearAlgebra
using Random
using Test
using StatsBase

@testset "DataNormalization" begin
    xs = randn(3, 100)

    @testset "No Normalization" begin
        normalizer = DataNormalization()
        transformation = StatsBase.fit(normalizer, xs)
        y = StatsBase.transform(transformation, xs)
        @test xs == y
    end

    @testset "ZScore" begin
        normalizer = DataNormalization(ZScoreTransform)
        transformation = StatsBase.fit(normalizer, xs)
        y = StatsBase.transform(transformation, xs)
        @test var(y) ≈ one(eltype(y)) atol = 1.0e-1
    end

    @testset "UnitRange" begin
        normalizer = DataNormalization(UnitRangeTransform)
        transformation = StatsBase.fit(normalizer, xs)
        y = StatsBase.transform(transformation, xs)
        @test all(zero(eltype(y)) .<= y .<= one(eltype(y)))
    end
end

@testset "DataProcessing" begin
    xs = randn(3, 100)
    ys = randn(2, 100)

    @testset "Splits" begin
        sampler = DataProcessing(split = 0.0)
        testdata, traindata = sampler(xs, ys)
        xtest, ytest = testdata
        @test size(xtest) == (3, 100)
        @test size(ytest) == (2, 100)

        sampler = DataProcessing(split = 1.0)
        testdata, traindata = sampler(xs, ys)
        xtest, ytest = testdata
        xtrain, ytrain = traindata.data

        @test size(xtest) == (3, 0)
        @test size(ytest) == (2, 0)

        @test size(xtrain) == (3, 100)
        @test size(ytrain) == (2, 100)

        sampler = DataProcessing(split = 0.8)
        testdata, traindata = sampler(xs, ys)
        xtest, ytest = testdata
        xtrain, ytrain = traindata.data

        @test size(xtest) == (3, 20)
        @test size(ytest) == (2, 20)

        @test size(xtrain) == (3, 80)
        @test size(ytrain) == (2, 80)
    end

    @testset "Batching" begin
        sampler = DataProcessing(split = 1.0, batchsize = 20)
        testdata, traindata = sampler(xs, ys)
        xtest, ytest = testdata

        @test size(xtest) == (3, 0)
        @test size(ytest) == (2, 0)
        @test length(traindata) == 5
        for (xtrain, ytrain) in traindata
            @test size(xtrain) == (3, 20)
            @test size(ytrain) == (2, 20)
        end

        sampler = DataProcessing(split = 1.0, batchsize = 23)
        testdata, traindata = sampler(xs, ys)
        xtest, ytest = testdata

        @test size(xtest) == (3, 0)
        @test size(ytest) == (2, 0)
        @test length(traindata) == 5
        for (i, (xtrain, ytrain)) in enumerate(traindata)
            if i < 5
                @test size(xtrain) == (3, 23)
                @test size(ytrain) == (2, 23)
            else
                @test size(xtrain) == (3, 8)
                @test size(ytrain) == (2, 8)
            end
        end
    end
end

@testset "CommonSolve Interface" begin
    # For internal testing only
    using DataDrivenDiffEq.CommonSolve

    struct DummyDataDrivenAlgorithm <: DataDrivenDiffEq.AbstractDataDrivenAlgorithm end
    struct DummyDataDrivenResult{IP} <: DataDrivenDiffEq.AbstractDataDrivenResult
        internal::IP
    end

    function CommonSolve.solve!(
            p::DataDrivenDiffEq.InternalDataDrivenProblem{
                DummyDataDrivenAlgorithm,
            }
        )
        return DummyDataDrivenResult(p)
    end

    x = rand(3, 100)
    y = rand(2, 100)
    u = rand(3, 100)

    @testset "Unit Basis" begin
        prob = DirectDataDrivenProblem(x, y)
        alg = DummyDataDrivenAlgorithm()
        @test_nowarn solve(prob, alg)
        res = solve(prob, alg)
        @test isa(res, DummyDataDrivenResult)
        @test length(res.internal.basis) == 3
        @test isempty(res.internal.implicit_idx)
        @test isempty(res.internal.control_idx)
    end

    @testset "Unit Basis controlled" begin
        prob = DirectDataDrivenProblem(x, y, U = u)
        alg = DummyDataDrivenAlgorithm()
        @test_nowarn solve(prob, alg)
        res = solve(prob, alg)
        @test isa(res, DummyDataDrivenResult)
        @test length(res.internal.basis) == 6
        @test isempty(res.internal.implicit_idx)
        @test res.internal.control_idx == vcat(zeros(Int, 3, 3), I(3))
    end

    @testset "Nonlinear Basis controlled" begin
        @variables xs[1:3]
        @variables us[1:3]
        xs = Symbolics.collect(xs)
        us = Symbolics.collect(us)
        basis = Basis([xs .* us[3]; xs[1] * sin(us[2]); xs[3] * us[1]], xs, controls = us)

        prob = DirectDataDrivenProblem(x, y, U = u)
        alg = DummyDataDrivenAlgorithm()
        @test_nowarn solve(prob, basis, alg)
        res = solve(prob, basis, alg)
        @test isa(res, DummyDataDrivenResult)
        @test length(res.internal.basis) == 5
        @test isempty(res.internal.implicit_idx)
        @test res.internal.control_idx == [0 0 1; 0 0 1; 0 0 1; 0 1 0; 1 0 0]
    end

    @testset "Nonlinear implicit basis controlled" begin
        @variables xs[1:3]
        @variables us[1:3]
        @variables ys[1:2]
        xs = Symbolics.collect(xs)
        us = Symbolics.collect(us)
        ys = Symbolics.collect(ys)
        basis = Basis(
            [
                xs .* us[3] .* exp(-ys[2]); xs[1] * sin(us[2]); xs[3] * us[1];
                sum(ys)
            ], xs,
            controls = us, implicits = ys
        )

        prob = DirectDataDrivenProblem(x, y, U = u)
        alg = DummyDataDrivenAlgorithm()
        @test_nowarn solve(prob, basis, alg)
        res = solve(prob, basis, alg)
        @test isa(res, DummyDataDrivenResult)
        @test length(res.internal.basis) == 6
        @test res.internal.control_idx == [0 0 1; 0 0 1; 0 0 1; 0 1 0; 1 0 0; 0 0 0]
        @test res.internal.implicit_idx == [0 1; 0 1; 0 1; 0 0; 0 0; 1 1]
    end
end
