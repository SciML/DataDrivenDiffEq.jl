using DataDrivenDMD
using LinearAlgebra: Eigen
using Test

@testset "AbstractKoopmanAlgorithm" begin
    X = [1.0 0.0 1.0 2.0; 0.0 1.0 1.0 3.0]
    Y = [2.0 1.0 3.0 5.0; 1.0 2.0 3.0 7.0]

    for algorithm in (DMDPINV(), DMDSVD(), TOTALDMD())
        @test algorithm isa DataDrivenDMD.AbstractKoopmanAlgorithm
        operator, inputmap = algorithm(X, Y)
        @test operator isa Eigen
        @test inputmap isa AbstractMatrix
    end
end
