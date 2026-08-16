using DataDrivenSparse
using Test

@testset "AbstractSparseRegressionAlgorithm" begin
    X = [1.0 2.0 3.0; 1.0 4.0 9.0]
    Y = [2.0 4.0 6.0]
    algorithm = STLSQ(0.1)

    @test algorithm isa AbstractSparseRegressionAlgorithm
    coefficients, thresholds, iterations = algorithm(X, Y)
    @test size(coefficients) == (size(Y, 1), size(X, 1))
    @test length(thresholds) == size(Y, 1)
    @test length(iterations) == size(Y, 1)
end

@testset "AbstractProximalOperator" begin
    @test SoftThreshold() isa AbstractProximalOperator
    x = [1.0, -2.0]
    SoftThreshold()(x, 0.5)
    @test x == [0.5, -1.5]
end
