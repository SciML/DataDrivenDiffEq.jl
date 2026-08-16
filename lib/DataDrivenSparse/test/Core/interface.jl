using DataDrivenDiffEq
using DataDrivenSparse
using Test

mutable struct InterfaceCache <: DataDrivenSparse.AbstractSparseRegressionCache
    Ã::Matrix{Float64}
    B̃::Vector{Float64}
    X::Matrix{Float64}
    X_prev::Matrix{Float64}
    active_set::BitMatrix
end

struct InterfaceSparseAlgorithm <: DataDrivenSparse.AbstractSparseRegressionAlgorithm
    thresholds::Vector{Float64}
end

struct InterfaceProximal <: DataDrivenSparse.AbstractProximalOperator end

function (::InterfaceProximal)(x::AbstractArray, lambda)
    x .= ifelse.(abs.(x) .> lambda, x, zero(eltype(x)))
    return x
end

function (::InterfaceProximal)(y::AbstractArray, x::AbstractArray, lambda)
    y .= ifelse.(abs.(x) .> lambda, x, zero(eltype(x)))
    return y
end

function DataDrivenSparse.active_set!(mask, ::InterfaceProximal, x, lambda)
    mask .= abs.(x) .> lambda
    return mask
end

DataDrivenSparse.get_thresholds(alg::InterfaceSparseAlgorithm) = alg.thresholds

function DataDrivenSparse.init_cache(
        ::InterfaceSparseAlgorithm, A::AbstractMatrix, b::AbstractVector
    )
    return InterfaceCache(
        Matrix{Float64}(A), Vector{Float64}(b), zeros(1, size(A, 1)),
        zeros(1, size(A, 1)), trues(1, size(A, 1))
    )
end

function DataDrivenSparse.step!(cache::InterfaceCache, lambda)
    cache.X_prev .= cache.X
    cache.X .= [2.0 0.0]
    cache.active_set .= abs.(cache.X) .> lambda
    return nothing
end

@testset "Generic sparse-regression interface" begin
    proximal = InterfaceProximal()
    x = [0.1, 2.0]
    y = similar(x)
    mask = falses(2)
    @test proximal(x, 1.0) == [0.0, 2.0]
    @test proximal(y, [0.1, 2.0], 1.0) == [0.0, 2.0]
    @test DataDrivenSparse.active_set!(mask, proximal, [0.1, 2.0], 1.0) ==
        BitVector([false, true])

    algorithm = InterfaceSparseAlgorithm([0.1, 0.5])
    options = DataDrivenCommonOptions(maxiters = 1, verbose = false)
    A = [1.0 2.0 3.0; 1.0 1.0 1.0]
    b = [2.0, 4.0, 6.0]

    cache = DataDrivenSparse.init_cache(algorithm, A, b)
    @test DataDrivenSparse.get_thresholds(algorithm) == [0.1, 0.5]
    @test DataDrivenSparse.step!(cache, 0.1) === nothing
    @test cache.X == [2.0 0.0]
    @test cache.active_set == BitMatrix([true false])

    solver = DataDrivenSparse.SparseLinearSolver(algorithm; options)
    results = solver(A, reshape(b, 1, :))
    @test length(results) == 1
    @test first(results)[1] isa InterfaceCache
end
