
@testset "Basis" begin
    @variables u[1:3]
    @parameters w[1:2]
    h = [u[1]; u[2]; cos(w[1]*u[2]+w[2]*u[3]); u[3]+u[2]]
    h_not_unique = [u[1]; u[1]; u[1]^1; h]
    basis = Basis(h_not_unique, u, parameters = w)

    @test isequal(variables(basis), u)
    @test isequal(parameters(basis), w)
    @test free_parameters(basis) == 6
    @test free_parameters(basis, operations = [+, cos]) == 7
    @test DataDrivenDiffEq.count_operation((ModelingToolkit.Constant(1) + cos(u[2])*sin(u[1]))^3, [+, cos, ^, *]) == 4

    basis_2 = unique(basis)
    @test isequal(basis, basis_2)
    @test size(basis) == size(h)
    @test basis([1.0; 2.0; π], p = [0. 1.]) ≈ [1.0; 2.0; -1.0; π+2.0]
    @test size(basis) == size(basis_2)
    push!(basis_2, sin(u[2]))
    @test size(basis_2)[1] == length(h)+1

    basis_3 = merge(basis, basis_2)
    @test size(basis_3) == size(basis_2)
    @test isequal(variables(basis_3), variables(basis_2))
    @test isequal(parameters(basis_3), parameters(basis_2))

    merge!(basis_3, basis)
    @test basis_3 == basis_2

    push!(basis, u[3]+u[2])
    unique!(basis)
    @test size(basis) == size(h)

    @variables a
    g = [u[1]; u[3]; a]
    basis = Basis(g, [u; a])
    @test basis([1; 2; 3; 4]) == [1; 3; 4]
    g = [u[1]; u[3]; u[2]]
    basis = Basis(g, u, parameters = [])
    X = ones(Float64, 3, 10)
    X[1, :] .= 3*X[1, :]
    X[3, :] .= 5*X[3, :]
    # Check the array evaluation
    @test basis(X) ≈ [1.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0] * X
    f = jacobian(basis)
    @test f([1;1;1], [0.0 0.0]) ≈ [1.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0]
    @test_nowarn sys = ODESystem(basis)
    @test_nowarn [xi for xi in basis]
    @test_nowarn basis[2:end]; basis[2]; first(basis); last(basis); basis[:]

    @variables u[1:2] t

    g = [u[2]; -sin(u[1])*exp(-t)]
    basis = Basis(g, [u...; t])
    @test_nowarn ODESystem(basis, t)
end
