using DataDrivenDiffEq
using ModelingToolkit

@testset "Basic Functionality" begin
    @variables x[1:3] u[1:2]
    @parameters p[1:2] t
    x = collect(x)
    u = collect(u)
    p = collect(p)

    g = [x[1] * p[1] + p[2] * x[2]; x[2] * u[1]; u[2] * x[3] + exp(-t)]

    @testset "Generated" begin
        b = Basis(g, x, parameters = p, controls = u, iv = t)

        true_res(x, p, t, u) = [sum(x[1:2] .* p); x[2] .* u[1]; u[2] .* x[3] .+ exp.(-t)]
        function true_res_(x, p, t)
            collect(hcat([true_res(x[:, i], p, t[i], zeros(2)) for i in 1:100]...))
        end

        function true_res_(x, p, t, u)
            collect(hcat([true_res(x[:, i], p, t[i], u[:, i]) for i in 1:100]...))
        end

        @testset "Vector evaluation" begin
            x0 = randn(3)
            p0 = randn(2)
            t0 = 0.0
            u0 = randn(2)

            @test isequal(b(x0), DataDrivenDiffEq.get_f(b)(x0, p, t, u))
            @test isequal(b(x0, p), DataDrivenDiffEq.get_f(b)(x0, p, t, u))
            @test isequal(b(x0, p, t), DataDrivenDiffEq.get_f(b)(x0, p, t, u))
            @test isequal(
                b(x0, p0, t, zeros(2)),
                DataDrivenDiffEq.get_f(b)(x0, p0, t, zeros(2))
            )
            @test isequal(
                b(x0, p0, t0, zeros(2)),
                DataDrivenDiffEq.get_f(b)(x0, p0, t0, zeros(2))
            )
            @test isequal(b(x0, p0, t0, u0), DataDrivenDiffEq.get_f(b)(x0, p0, t0, u0))
        end

        @testset "Array evaluation" begin
            # Array call
            x0 = randn(3, 100)
            p0 = randn(2)
            t0 = randn(100)
            u0 = randn(2, 100)

            # These first two fail, since exp(-t) != exp(getindex(t,1))
            @test isequal(b(x0, p0, t0, u0), true_res_(x0, p0, t0, u0))
        end
    end

    @testset "Evaluated" begin
        b = Basis(g, x, parameters = p, controls = u, iv = t, eval_expression = false)

        true_res(x, p, t, u) = [sum(x[1:2] .* p); x[2] .* u[1]; u[2] .* x[3] .+ exp.(-t)]
        function true_res_(x, p, t)
            collect(hcat([true_res(x[:, i], p, t[i], zeros(2)) for i in 1:100]...))
        end

        function true_res_(x, p, t, u)
            collect(hcat([true_res(x[:, i], p, t[i], u[:, i]) for i in 1:100]...))
        end

        @testset "Vector evaluation" begin
            x0 = randn(3)
            p0 = randn(2)
            t0 = 0.0
            u0 = randn(2)

            @test isequal(b(x0), DataDrivenDiffEq.get_f(b)(x0, p, t, u))
            @test isequal(b(x0, p), DataDrivenDiffEq.get_f(b)(x0, p, t, u))
            @test isequal(b(x0, p, t), DataDrivenDiffEq.get_f(b)(x0, p, t, u))
            @test isequal(
                b(x0, p0, t, zeros(2)),
                DataDrivenDiffEq.get_f(b)(x0, p0, t, zeros(2))
            )
            @test isequal(
                b(x0, p0, t0, zeros(2)),
                DataDrivenDiffEq.get_f(b)(x0, p0, t0, zeros(2))
            )
            @test isequal(b(x0, p0, t0, u0), DataDrivenDiffEq.get_f(b)(x0, p0, t0, u0))
        end

        @testset "Array evaluation" begin
            # Array call
            x0 = randn(3, 100)
            p0 = randn(2)
            t0 = randn(100)
            u0 = randn(2, 100)

            # These first two fail, since exp(-t) != exp(getindex(t,1))
            @test isequal(b(x0, p0, t0, u0), true_res_(x0, p0, t0, u0))
        end
    end
end

@testset "Basis Manipulation" begin
    @parameters w[1:2] t
    @variables u(t)[1:3]

    h = [u; cos(w[1] * u[2] + w[2] * u[3]); 5 * u[3] + u[2]]
    h_not_unique = [u[1]; u[1]; u[1]^1; h; 1]
    basis = Basis(h_not_unique, u, parameters = w, iv = t)
    basis_2 = Basis(h_not_unique, u, parameters = w, iv = t, linear_independent = true)

    # Check getters
    @test isequal(states(basis), u)
    @test isequal(parameters(basis), w)
    @test isequal(ModelingToolkit.get_iv(basis), t)
    @test isequal(controls(basis), [])
    @test !DataDrivenDiffEq.is_implicit(basis)
    @test DataDrivenDiffEq.count_operation(
        (1 + cos(u[2]) * sin(u[1]))^3,
        [+, cos, ^, *]
    ) == 4

    # Check array functionalities
    unique!(basis)

    @test size(basis) == (6,)
    @test size(basis_2) == (5,)
    # Note: Order may differ due to internal Symbolics representation
    # The linear_independent basis extracts terms which may be reordered
    # Use sorted comparison to handle non-deterministic ordering
    @test sort(basis_2([1.0; 2.0; π], [0.0; 1.0])) ≈ sort([1.0; -1.0; π; 2.0; 1.0])
    @test sort(basis([1.0; 2.0; π], [0.0; 1.0])) ≈ sort([1.0; 2.0; π; -1.0; 5 * π + 2.0; 1.0])

    @test size(basis) == size(basis_2) .+ (1,)
    push!(basis_2, sin(u[2]))
    @test size(basis_2) == (6,)
    basis_3 = merge(basis, basis_2)
    @test size(basis_3) == (7,)
    @test isequal(states(basis_3), states(basis_2))
    @test isequal(parameters(basis_3), parameters(basis_2))
    merge!(basis_3, basis)
    push!(basis, 5 * u[3] + u[2])
    unique!(basis) # Does not remove
    @test size(basis) == (6,)
end

@testset "Utils and calls" begin
    @variables a
    @variables u[1:3]
    @parameters w[1:2] t
    g = [u[1]; u[3]; a]
    basis = Basis(g, [u; a])
    @test basis([1; 2; 3; 4]) == [1; 3; 4]
    g = [u[1]; u[3]; u[2]]
    basis = Basis(g, u, parameters = [])
    X = ones(Float64, 3, 10)
    X[1, :] .= 3 * X[1, :]
    X[3, :] .= 5 * X[3, :]
    # Check the array evaluation
    @test basis(X, [], zeros(10)) ≈ [1.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0] * X
    Y = similar(X)
    basis(Y, X, [], zeros(10))
    @test Y ≈ [1.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0] * X
    f = jacobian(basis)
    @test f([1.0; 1.0; 1.0], [0.0; 0.0], 0.0) ≈ [1.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0]
    @test_nowarn [xi for xi in basis]
    @test_nowarn basis[2:end]

    @variables u[1:3] t

    g = [u[2]; -sin(u[1]) * exp(-t); u[2] + u[3]]
    basis = Basis(g, u, iv = t)

    f_(u, p, t) = [u[3]; u[2] * u[1]; p[1] * sin(u[1]) * u[2]; p[2] * t]
    b = Basis(f_, u, parameters = w, iv = t)
    # Default values
    @test iszero(get_parameter_values(b))
    @test f_([1; 2; 3], [2; 0], 3.0) ≈ b([1; 2; 3], [2; 0], 3.0)

    @parameters w[1:2] = [1.0; 2.0]
    w = collect(w)
    b = Basis(f_, u, parameters = w, iv = t)
    # Default values
    @test get_parameter_values(b) == [1.0; 2.0]
    @test last.(get_parameter_map(b)) == [1.0; 2.0]
end

@testset "ODEProblem from Basis (Issue #559)" begin
    # Regression test for issue #559: solve throws MethodError when creating
    # ODEProblem from Basis due to symbolic to numeric conversion issues
    using OrdinaryDiffEqTsit5

    # Create a simple basis with parameters that have no default values
    @variables u[1:2]
    @parameters w[1:2]
    u = collect(u)
    w = collect(w)

    # Create a basis with parameters without default values
    # This tests the zero(Symbolics.symtype(p)) code path
    h = [u[1]^2 + w[1] * u[2]; sin(w[2] * u[1])]
    basis = Basis(h, u, parameters = w)

    # Test that get_parameter_values returns unwrapped numeric values, not symbolic
    params = get_parameter_values(basis)
    @test params isa Vector
    @test all(p -> !(p isa Num), params)  # Should not be Num/symbolic
    @test all(iszero, params)  # Parameters without defaults should be zero

    # Test that get_parameter_map also returns unwrapped numeric values
    param_map = get_parameter_map(basis)
    @test all(pair -> !(last(pair) isa Num), param_map)

    # Test that we can create an ODEProblem from the basis
    # This is the key test from issue #559 - should not throw MethodError
    # about "Cannot convert BasicSymbolic{Real} to Float64"
    u0 = [1.0, 2.0]
    tspan = (0.0, 0.1)  # Very short timespan
    p_values = [0.01, 0.01]  # Very small parameter values
    recovered_model = ODEProblem(basis, u0, tspan, p_values)
    @test recovered_model isa ODEProblem

    # Test that we can initialize the integrator without the symbolic conversion error
    sol = solve(recovered_model, Tsit5(), save_everystep = false)

    # Also test with parameters that have default values
    @parameters w2[1:2] = [1.5, 2.5]
    w2 = collect(w2)
    h2 = [u[1]^2 + w2[1] * u[2]; sin(w2[2] * u[1])]
    basis2 = Basis(h2, u, parameters = w2)

    # Test that get_parameter_values returns the default values unwrapped
    params2 = get_parameter_values(basis2)
    @test all(p -> !(p isa Num), params2)
    @test params2 ≈ [1.5, 2.5]

    # Test creating ODEProblem with default parameter values
    recovered_model2 = ODEProblem(basis2, u0, tspan, params2)
    @test recovered_model2 isa ODEProblem
end
