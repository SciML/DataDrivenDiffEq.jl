@info "Starting standard basis tests"
@testset "Basis" begin
    @variables u[1:3]
    @parameters w[1:2] t
    h = [u[1]; u[2]; cos(w[1] * u[2] + w[2] * u[3]); u[3] + u[2]]
    h_not_unique = [u[1]; u[1]; u[1]^1; h]
    basis = Basis(h_not_unique, u, parameters=w, iv=t)
    basis_2 = Basis(h_not_unique, u, parameters=w, iv=t, linear_independent=true)
    @test isequal(states(basis), u)
    @test isequal(parameters(basis), w)
    @test isequal(independent_variable(basis), t)

    @test free_parameters(basis) == 6
    @test free_parameters(basis_2) == 5
    @test free_parameters(basis, operations=[+, cos]) == 7
    @test free_parameters(basis_2, operations=[+, cos]) == 6
    @test DataDrivenDiffEq.count_operation((1 + cos(u[2]) * sin(u[1]))^3, [+, cos, ^, *]) == 4

    basis_2 = unique(basis)
    @test isequal(basis, basis_2)
    @test size(basis) == size(h)
    @test basis([1.0; 2.0; π], [0. 1.]) ≈ [1.0; 2.0; -1.0; π + 2.0]
    @test size(basis) == size(basis_2)
    push!(basis_2, sin(u[2]))
    @test size(basis_2)[1] == length(h) + 1

    basis_3 = merge(basis, basis_2)
    @test size(basis_3) == size(basis_2)
    @test isequal(states(basis_3), states(basis_2))
    @test isequal(parameters(basis_3), parameters(basis_2))

    merge!(basis_3, basis)
    @test basis_3 == basis_2

    push!(basis, u[3] + u[2])
    unique!(basis)
    @test size(basis) == size(h)

    @variables a
    g = [u[1]; u[3]; a]
    basis = Basis(g, [u; a])
    @test basis([1; 2; 3; 4]) == [1; 3; 4]
    g = [u[1]; u[3]; u[2]]
    basis = Basis(g, u, parameters=[])
    X = ones(Float64, 3, 10)
    X[1, :] .= 3 * X[1, :]
    X[3, :] .= 5 * X[3, :]
    # Check the array evaluation
    @test basis(X) ≈ [1.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0] * X
    Y = similar(X)
    basis(Y, X)
    @test Y ≈ [1.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0] * X
    f = jacobian(basis)
    @test f([1;1;1], [0.0 0.0], 0.0) ≈ [1.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0]
    @test_nowarn sys = ODESystem(basis)
    @test_nowarn [xi for xi in basis]
    @test_nowarn basis[2:end]; basis[2]; first(basis); last(basis); basis[:]

    @variables u[1:3] t

    g = [u[2]; -sin(u[1]) * exp(-t); u[2] + u[3]]
    basis = Basis(g, u, iv=t)
    # @test_nowarn ODESystem(basis)

    f(u, p, t) = [u[3]; u[2] * u[1]; p[1] * sin(u[1]) * u[2]; p[2] * t]
    b = Basis(f, u, parameters=w, iv=t)
    @test f([1; 2; 3], [2; 0], 3.0) ≈ b([1; 2; 3], [2; 0], 3.0)
    @test_throws AssertionError ODESystem(b)
end

# @testset "Basis Generators" begin
#    @variables u[1:3]
#    @test all(isequal.(sin_basis(u, 1), sin.(1 .* u)))
#    @test all(isequal.(cos_basis(u, 2), vcat(cos.(1 .* u), cos.(2 .*u))))
#    @test all(isequal.(chebyshev_basis(u, 1), cos.( 1 .* acos.(u))))
#    @test all(isequal.(fourier_basis(u, 1), sin.(1 .* u ./2)))
#    @test all(isequal.(monomial_basis(u, 1), u.^1))
#    @test all(isequal.(polynomial_basis(u, 2), [ModelingToolkit.Constant(1); u[1]^1; u[1]^2; u[2]^1; u[1]^1*u[2]^1; u[2]^2; u[3]^1; u[1]^1*u[3]^1; u[2]^1*u[3]^1; u[3]^2]))
#    
#    @test all(isequal.(sin_basis(u, 1:2), vcat([sin.(i .* u) for i in 1:2]...)))
#    @test all(isequal.(cos_basis(u, 1:5), vcat([cos.(i .* u) for i in 1:5]...)))
#    @test all(isequal.(chebyshev_basis(u, [1;2]), vcat([cos.(i .* acos.(u)) for i in 1:2]...)))
#    @test all(isequal.(fourier_basis(u, 1:2), vcat(sin.(1 .* u ./2), cos.( 2 .* u ./ 2))))
# end

