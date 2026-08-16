using DataDrivenDMD
using LinearAlgebra: Eigen
using Test

@testset "AbstractKoopmanAlgorithm" begin
    X = [1.0 2.0; 2.0 4.0]
    Y = [2.0 4.0; 4.0 8.0]

    for algorithm in (DMDPINV(), DMDSVD(), TOTALDMD())
        @test algorithm isa AbstractKoopmanAlgorithm
        operator, inputmap = algorithm(X, Y)
        @test operator isa Eigen
        @test inputmap isa AbstractMatrix
    end
end
