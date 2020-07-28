
@info "Starting SINDy tests"

@testset "SInDy" begin
    # Create a nonlinear pendulum
    function pendulum(u, p, t)
        x = u[2]
        y = -9.81sin(u[1]) - 0.1u[2]^3 -0.2*cos(u[1])
        return [x;y]
    end

    u0 = [0.99π; -1.0]
    tspan = (0.0, 20.0)
    dt = 0.3
    prob = ODEProblem(pendulum, u0, tspan)
    sol = solve(prob, Tsit5(), saveat = dt)


    # Create the differential data
    DX = similar(sol[:,:])
    for (i, xi) in enumerate(eachcol(sol[:,:]))
        DX[:,i] = pendulum(xi, [], 0.0)
    end

    # Create a basis
    @variables u[1:2]
    # Lots of polynomials
    polys = Operation[1]
    for i ∈ 1:5
        push!(polys, u.^i...)
        for j ∈ 1:i-1
            push!(polys, u[1]^i*u[2]^j)
        end
    end

    # And some other stuff
    h = [cos(u[1]); sin(u[1]); u[1]*u[2]; u[1]*sin(u[2]); u[2]*cos(u[2]); polys]

    basis = Basis(h, u)

    opt = STRRidge(1e-2)
    Ψ = SInDy(sol[:,:], DX, basis, opt = opt, maxiter = 2000)
    @test_nowarn set_threshold!(opt, 1e-2)
    
    @test length(Ψ) == 2
    @test size(Ψ) == (2, )
    @test parameters(Ψ) ≈ [1.0; -0.2; -9.81; -0.1]
    W = get_coefficients(Ψ)
    @test size(W) == (length(basis), 2)
    @test_nowarn get_aicc(Ψ)
    @test sum(get_sparsity(Ψ)) == 4
    @test sum(get_error(Ψ)) < 1e-10
    @test Ψ(u0) ≈ pendulum(u0, nothing, nothing)

    # Simulate
    estimator = ODEProblem(dynamics(Ψ), u0, tspan, parameters(Ψ))
    sol_ = solve(estimator,Tsit5(), saveat = dt)
    @test sol[:,:] ≈ sol_[:,:]

    opt = ADMM(1e-2, 0.7)
    Ψ = SInDy(sol[:,:], DX, basis, maxiter = 5000, opt = opt)
    @test_nowarn set_threshold!(opt, 1e-2)

    # Simulate
    estimator = ODEProblem(dynamics(Ψ), u0, tspan, parameters(Ψ))
    sol_2 = solve(estimator,Tsit5(), saveat = dt)
    @test norm(sol[:,:] - sol_2[:,:], 2) < 2e-1

    opt = SR3(1e-2, 1.0)
    Ψ = SInDy(sol[:,:], DX, basis, maxiter = 5000, opt = opt)
    @test_nowarn set_threshold!(opt, 0.1)

    # Simulate
    estimator = ODEProblem(dynamics(Ψ), u0, tspan, parameters(Ψ))
    sol_3 = solve(estimator,Tsit5(), saveat = dt)
    @test norm(sol[:,:] - sol_3[:,:], 2) < 1e-1

    # Now use the threshold adaptation
    opt = STRRidge(1e-2)
    λs = exp10.(-7:0.1:-1)
    for alg in [WeightedSum(), WeightedExponentialSum(), GoalProgramming()]
        Ψ = SInDy(sol[:,:], DX[:, :], basis, λs, alg = alg, maxiter = 100, opt = opt)
        estimator = ODEProblem(dynamics(Ψ), u0, tspan, parameters(Ψ))
        sol_4 = solve(estimator, Tsit5(), saveat = dt)
        @test norm(sol[:,:] - sol_4[:,:], 2) < 1e-1
    end

    # Check for errors
    @test_nowarn SInDy(sol[:,:], DX[1,:], basis, λs, maxiter = 1, opt = opt)
    @test_nowarn SInDy(sol[:, :], DX[1, :], basis, λs, maxiter = 1, opt = opt, denoise = true, normalize = true)

    # Check with noise
    X = sol[:, :] + 1e-3*randn(size(sol[:,:])...)
    opt = SR3(1e-1, 10.0)
    Ψ = SInDy(X, DX, basis, λs, maxiter = 10000, opt = opt, denoise = true, normalize = true)

    estimator = ODEProblem(dynamics(Ψ), u0, tspan, parameters(Ψ))
    sol_5 = solve(estimator,Tsit5(), saveat = dt)
    @test norm(sol[:,:] - sol_5[:,:], 2) < 5e-1

    # Check sparse_regression
    X .= Array(sol)
    opt = SR3(1e-1, 1.0)
    maxiter = 5000
    θ = basis(X)
    Ξ1, iters_1 = sparse_regression(X, DX, basis, parameters(basis), [], maxiter, opt, true, true, eps())
    Ξ2 = similar(Ξ1)
    Ξ3 = similar(Ξ1)
    iters_2 = sparse_regression!(Ξ2, X, DX, basis, parameters(basis), [], maxiter, opt, true, true, eps())
    iters_3 = sparse_regression!(Ξ3, θ, DX, maxiter, opt, true, true, eps())

    @test iters_1 == iters_2 == iters_3
    @test Ξ1 ≈ Ξ2 ≈ Ξ3
    @test isapprox(norm(Ξ1'*θ - DX,2), 10.18; atol = 1e-1)
end
