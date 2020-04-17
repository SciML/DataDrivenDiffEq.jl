
@testset "DMD" begin
    # Create some linear data
    A = [0.9 -0.2; 0.0 0.2]
    y = [[10.; -10.]]
    for i in 1:10
        push!(y, A*y[end])
    end
    X = hcat(y...)

    estimator = DMD(X)
    # TODO export
    #@test isstable(estimator)
    @test operator(estimator) ≈ A
    @test eigvals(estimator) ≈ eigvals(A)
    @test eigvecs(estimator) ≈ eigvecs(A)
    @test is_discrete(estimator)
    # TODO add Array{T,2}
    #@test estimator(X[:, 1]) ≈ X[:, 2]
    @test_nowarn update!(estimator, X[:, end-1], X[:,end])
    @test_throws AssertionError outputmap(estimator)
end

@testset "EDMD" begin
    # Test for linear system
    function linear_sys(u, p, t)
        x = -0.9*u[1]
        y = -0.3*u[2]
        return [x;y]
    end

    u0 = [π; 1.0]
    tspan = (0.0, 20.0)
    prob = DiscreteProblem(linear_sys, u0, tspan)
    sol = solve(prob,FunctionMap())

    @variables u[1:2]
    h = [1u[1]; 1u[2]; sin(u[1]); cos(u[1]); u[1]*u[2]]
    basis = Basis(h, u)

    estimator = EDMD(sol[:,:], basis, alg = DMDSVD())
    @test basis == estimator.basis
    basis_2 = reduce_basis(estimator, threshold = 1e-5)
    @test size(basis_2)[1] < size(basis)[1]
    estimator_2 = EDMD(sol[:,:], basis_2)
    p1 = DiscreteProblem(estimator, u0, tspan)
    s1 = solve(p1,FunctionMap())
    p2 = DiscreteProblem(estimator_2, u0, tspan)
    s2 = solve(p2,FunctionMap())
    # TODO add linear dynamics ?
    #p3 = DiscreteProblem(linear_dynamics(estimator_2), estimator_2(u0), tspan, [])
    #s3 = solve(p3,FunctionMap())
    @test sol[:,:] ≈ s1[:,:]
    @test sol[:,:] ≈ s2[:,:]
    @test eigvals(estimator_2) ≈ [-0.9; -0.3]
    @test isstable(estimator_2)

    # Test for nonlinear system
    function nonlinear_sys(du, u, p, t)
        du[1] = sin(u[1])
        du[2] = -0.3*u[2] -0.9*u[1]
    end

    prob = DiscreteProblem(nonlinear_sys, u0, tspan)
    sol = solve(prob,FunctionMap())
    estimator = EDMD(sol[:,:], basis, alg = DMDPINV())
    p4 = DiscreteProblem(estimator, u0, tspan)
    s4 = solve(p4,FunctionMap())
    @test sol[:,:] ≈ s4[:,:]
end


@testset "DMDc" begin
    # Define measurements from unstable system with known control input
    X = [4 2 1 0.5 0.25; 7 0.7 0.07 0.007 0.0007]
    U = [-4 -2 -1 -0.5]
    B = Float32[1; 0]

    # But with a little more knowledge
    sys = DMDc(X, U, B = B)
    @test operator(sys) ≈[1.5 0; 0 0.1]
    @test inputmap(sys) ≈ [1.0; 0.0]
    @test !isstable(sys)
    @test_nowarn eigen(sys)
    sys2 = DMDc(X, U, B = B, alg = DMDSVD())
    @test operator(sys2) ≈ operator(sys)
end
