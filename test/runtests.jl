using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra
using Test

@testset "Basis" begin
    @variables u[1:3]
    @parameters w[1:2]
    h = [1u[1]; 1u[2]; cos(w[1]*u[2]+w[2]*u[3]); 1u[3]+1u[2]]
    basis = Basis(h, u, parameters = w)
    basis_2 = unique(basis)
    @test size(basis) == size(h)
    # TODO This works fine for "manual" execution of the testset but fails
    # with Pkg.test
    @test basis([1.0; 2.0; π], p = [0. 1.]) ≈ [1.0; 2.0; -1.0; π+2.0]
    @test size(basis) == size(basis_2)
    push!(basis_2, sin(u[2]))
    @test size(basis_2)[1] == length(h)+1
    push!(basis, 1u[3]+1u[2])
    unique!(basis)
    @test size(basis) == size(h)
    g = [1.0*u[1]; 1.0*u[3]; 1.0*u[2]]
    basis = Basis(g, u, parameters = w)
    f = jacobian(basis)
    @test f([1;1;1], [0.0 0.0]) ≈ [1.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0]
end

@testset "DMD" begin
    # Create some linear data
    A = [0.9 -0.2; 0.0 0.2]
    y = [[10.; -10.]]
    for i in 1:10
        push!(y, A*y[end])
    end
    X = hcat(y...)
    estimator = ExactDMD(X[:,1:end-2])
    @test isstable(estimator)
    @test estimator.Ã ≈ A
    @test eigvals(estimator) ≈ eigvals(A)
    @test eigvecs(estimator) ≈ eigvecs(A)
    @test_nowarn dynamics(estimator)
    @test_throws AssertionError dynamics(estimator, discrete = false)
    @test_nowarn update!(estimator, X[:, end-1], X[:,end])
end




@testset "EDMD" begin
end

@testset "SInDy" begin
end
