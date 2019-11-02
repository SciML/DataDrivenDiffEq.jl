using DataDrivenDiffEq
using ModelingToolkit
using Test

@testset "Basis" begin
    @variables u[1:3]
    @parameters w[1:2]
    h = [1.0*u[1]; 2.0*u[2]; cos(w[1]*u[2]+w[2]*u[3]); u[3]+u[2]]

    basis = Basis(h)
    basis_2 = unique(basis)
    @test size(basis) == size(h)
    @test size(basis) == size(basis_2)
    push!(basis_2, sin(u[2]))
    @test size(basis_2)[1] == length(h)+1
    push!(basis, u[3]+u[2])
    unique!(basis)
    @test size(basis) == size(h)
    @test basis([1; 2; π], p = [0 1]) == [1.0; 4.0; -1.0; π+2.0]
    g = [1.0*u[1]; 1.0*u[3]; 1.0*u[2]]
    basis = Basis(g)
    f = jacobian(basis, u, w)
    @test f([1;1;1], [0.0 0.0]) ≈ [1.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0]
end
