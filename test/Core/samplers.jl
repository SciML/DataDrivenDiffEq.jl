using DataDrivenDiffEq
using LinearAlgebra

# Random Test Problem of size 100
X = randn(2, 100)
Y = randn(1, 100)

prob = DirectDataDrivenProblem(X, Y)

@testset "Train Test Split" begin
    for r in 0.2:0.2:0.8
        split = Split(ratio = r)
        train, test = split(prob)
        @test length(train) == Int(r * 100)
        @test length(test) == Int(100 - r * 100)
        @test length(train) + length(test) == 100
    end
end

@testset "Mini Batching" begin
    for n in 2:5, repeat_ in [true, false],
            shuffle_ in [true, false]
        batch = Batcher(n = n, repeated = repeat_, shuffle = shuffle_)
        train, test = batch(prob)
        @test length(train) == n
        @test all(length.(train) .>= round(Int, 100 / n))
        @test sum(length.(train)) == 100
    end
end

@testset "DataSampler" begin
    for repeat_ in [true, false], shuffle_ in [true, false]

        ds = DataSampler(
            Batcher(n = 2, repeated = repeat_, shuffle = shuffle_),
            Split(ratio = 0.8)
        )
        train, test = ds(prob)
        @test length(test) == 20
        @test sum(length, train) == 80
    end
end
