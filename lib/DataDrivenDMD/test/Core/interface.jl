using DataDrivenDMD
using LinearAlgebra: I, eigen
using Test

struct InterfaceKoopman <: DataDrivenDMD.AbstractKoopmanAlgorithm end

function (::InterfaceKoopman)(X::AbstractMatrix, Y::AbstractMatrix)
    return eigen(Y / X), zeros(eltype(X), size(Y, 1), 0)
end

function (::InterfaceKoopman)(
        X::AbstractMatrix, Y::AbstractMatrix, U::AbstractMatrix
    )
    return eigen(Y / X), zeros(eltype(X), size(Y, 1), size(U, 1))
end

@testset "Generic Koopman algorithm interface" begin
    X = Matrix{Float64}(I, 2, 2)
    Y = [2.0 0.0; 0.0 3.0]
    U = zeros(1, 2)
    B = zeros(2, 1)
    algorithm = InterfaceKoopman()

    K, B0 = algorithm(X, Y)
    @test Matrix(K) == Y
    @test isempty(B0)

    K, B1 = algorithm(X, Y, U)
    @test Matrix(K) == Y
    @test size(B1) == (2, 1)

    K, B2 = algorithm(X, Y, U, B)
    @test Matrix(K) == Y
    @test B2 === B

    K, B3 = algorithm(X, Y, U, nothing)
    @test Matrix(K) == Y
    @test size(B3) == (2, 1)
end
