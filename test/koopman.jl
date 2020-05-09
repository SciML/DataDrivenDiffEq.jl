
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

    @variables u[1:2]
    b = Basis(u, u)
    sys = EDMD(X[:, 1:3], b)
    @test_nowarn update!(sys, X[:, 3:end-1], X[:, 4:end], threshold = 1e-17)
    @test operator(sys) ≈ A

    # Test for a large system with reduced svd
    dd(u, p, t) = -0.9*ones(length(u))*u[1] - 0.05*u
    u0 = randn(100)*100.0
    u0[1] = 100.0
    prob = ODEProblem(dd, u0, (0.0, 100.0))
    sol = solve(prob, Tsit5(), saveat = 0.1)

    X = sol[:,:]
    DX = sol(sol.t, Val{1})[:,:]

    t = DMDSVD(0.01)
    d = gDMD(X, DX, alg = t)

    test = ODEProblem(d, u0, (0.0, 100.0))
    sol_ = solve(test, Tsit5(), saveat = 0.1)
    @test norm(sol-sol_, Inf) < 2.0

    d2 = gDMD(sol.t, X)
    d3 = gDMD(sol.t, X, dt = 0.2)
    @test generator(d) ≈ generator(d2) atol = 1e-1
    @test generator(d) ≈ generator(d3) atol = 1e-1

    K = -0.5*I + [0 0 -0.2; 0.1 0 -0.1; 0. -0.2 0]
    F = qr(randn(20, 3))
    Q = F.Q[:, 1:3]
    dudt(u, p, t) = K*u
    prob = ODEProblem(dudt, [10.0; 0.3; -5.0], (0.0, 10.0))
    sol_ = solve(prob, Tsit5(), saveat = 0.01)
    alg = TOTALDMD(3, DMDPINV())
    X = Q*sol_[:,:] + 1e-3*randn(20, 1001)
    DX = Q*sol_(sol_.t, Val{1})[:,:] + 1e-3*randn(20, 1001)
    approx = gDMD(X, DX, alg = alg)
    @test Q'*generator(approx)*Q ≈ K atol = 1e-1
    @test Q*K*Q' ≈ generator(approx) atol = 1e-1
    alg = TOTALDMD(0.01, DMDSVD())
    approx = gDMD(X, DX, alg = alg)
    @test Q'*generator(approx)*Q ≈ K atol = 1e-1
    @test Q*K*Q' ≈ generator(approx) atol = 1e-1

end

@testset "EDMD" begin
    # Test for linear system
    function linear_sys(u, p, t)
        x = -0.9*u[1]
        y = -0.3*u[2]
        return [x;y]
    end

    u0 = [π; 1.0]
    tspan = (0.0, 50.0)
    prob = DiscreteProblem(linear_sys, u0, tspan)
    sol = solve(prob,FunctionMap())

    @variables u[1:2]
    h = [u[1]; u[2]; sin(u[1]); cos(u[1]); u[1]*u[2]; u[2]^2]
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
        du[1] = 0.9u[1] + 0.1u[2]^2
        du[2] = sin(u[1]) - 0.1u[1]
    end

    prob = DiscreteProblem(nonlinear_sys, u0, tspan)
    sol = solve(prob,FunctionMap())
    estimator = EDMD(sol[:,:], basis, alg = DMDPINV())
    sys = ODESystem(estimator)
    #@test isa(sys, ODESystem)
    dudt = ODEFunction(sys)
    p4 = DiscreteProblem(dudt, u0, tspan)

    s4 = solve(p4,FunctionMap())
    @test sol[:,:] ≈ s4[:,:]


    function slow_manifold(du, u, p, t)
      du[1] = p[1]*u[1]
      du[2] = p[2]*(u[2]-u[1]^2)
    end

    u0 = [3.0; -2.0]
    tspan = (0.0, 10.0)
    p = [-0.05, -0.9]

    problem = ODEProblem(slow_manifold, u0, tspan, p)
    sol_slow = solve(problem, Tsit5(), saveat = 0.2)

    X = Array(sol_slow)
    # This enforces more accurate results
    DX = similar(X)
    for (i,dx) in enumerate(eachcol(DX))
        slow_manifold(dx, X[:, i], p, sol_slow.t[i])
    end
    basis = Basis([u; u[1]^2], u)
    estimator2 = gEDMD(X, DX, basis)
    A_analytical = [p[1] 0 0; 0 p[2] -p[2]; 0 0 2*p[1]]
    outputmap(estimator2)
    generator(estimator2)
    @test generator(estimator2) ≈ [p[1] 0 0; 0 p[2] -p[2]; 0 0 2*p[1]] atol =  1e-3
    @test outputmap(estimator2) ≈ [1 0 0 ; 0 1 0] atol = 1e-3
    @test abs.(modes(estimator2)) ≈ abs.(eigvecs(A_analytical)) atol = 1e-3
    @test frequencies(estimator2) ≈ eigvals(A_analytical) atol = 1e-3


    function nonlinear_sys2(du, u, p, t)
        du[1] = u[2]
        du[2] = -0.9u[1]
    end

    prob_nl = ODEProblem(nonlinear_sys2, u0, (0.0, 10.0))
    sol_nl = solve(prob_nl, Tsit5())

    DX = sol_nl(sol_nl.t, Val{1})[:,:]
    X = Array(sol_nl)

    basis = Basis(u, u)
    estimator3 = gEDMD(X, DX, basis)
    @test generator(estimator3) ≈ [0  1.0; -0.9 0]
    @test outputmap(estimator3) ≈ I(2)

    d2 = gEDMD(sol_nl.t, X, basis)
    d3 = gEDMD(sol_nl.t, X, basis, dt = 0.1)
    @test generator(estimator3) ≈ generator(d2) atol = 1e-1
    @test generator(estimator3) ≈ generator(d3) atol = 1e-1

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
    sys3 = gDMDc(X[:, 1:end-1], X[:, 2:end], U, B = B)
    @test generator(sys3) ≈ [1.5 0; 0 0.1]

    A = [0.9 0.3; -0.2 0.9]
    B = [1.0 0;0 0.5]
    X = zeros(Float64, 2, 11)
    U = zeros(Float64, 2, 10)
    X[:, 1] = [1.0; -3.0]
    for i in 1:10
        U[:, i] .= [sin(2.0/5.0*i); cos(5.0/7.0*i)]
        X[:, i+1] .= A*X[:,i]+B*[sin(2.0/5.0*i); cos(5.0/7.0*i)]
    end

    sys4 = DMDc(X, U)
    @test operator(sys4) ≈ A
    @test inputmap(sys4) ≈ B
    sys5 = gDMDc(X[:, 1:end-1], X[:, 2:end], U[:, 1:end])
    @test_throws AssertionError operator(sys5)
    @test generator(sys5) ≈ A
    @test inputmap(sys5) ≈ B

    sys6 = gDMDc(collect(0.0:9.0), X[:, 1:end-1], U, B = B)
    @test exp(generator(sys6)) ≈ A atol = 1e-1
    @test inputmap(sys6) ≈ B

end
