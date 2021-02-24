
@testset "SINDy" begin
    using Random

    Random.seed!(5723)

    function pendulum(u, p, t)
        x = u[2]
        y = -9.81sin(u[1]) - 0.1u[2]^3 - 0.2 * cos(u[1])
        return [x;y]
    end

    u0 = [0.99π; -1.0]
    tspan = (0.0, 20.0)
    dt = 0.3
    prob = ODEProblem(pendulum, u0, tspan)
    sol = solve(prob, Tsit5(), saveat=dt)


    # Create the differential data
    X = Array(sol)
    Xₙ = X + 1e-1*randn(size(X))
    DX = similar(sol[:,:])
    DXₙ = similar(Xₙ)

    for (i, xi) in enumerate(eachcol(sol[:,:]))
        DX[:,i] = pendulum(xi, [], 0.0)
        DXₙ[:,i] = pendulum(Xₙ[:,i], [], 0.0)
    end

    # Smaller dataset
    Xₛ = X[:, 1:20]
    DXₛ = DX[:, 1:20]

    # Create a basis
    @variables u[1:2]
    polys = polynomial_basis(u, 5)
    h = [cos(u[1]); sin(u[1]); u[1] * sin(u[2]); u[2] * cos(u[2]); polys]
    basis = Basis(h, u)

    ## Check conversions
    @testset "General Tests" begin

        Ψ = SINDy(X, DX, basis, STLSQ(1e-2), maxiter=100, denoise=false, normalize=false)
        sys = ODESystem(Ψ)
        dudt = ODEFunction(sys)
        prob = ODEProblem(dudt, u0, tspan, parameters(Ψ))
        sol_ = solve(prob, Tsit5(), saveat=dt)
        @test isapprox(opnorm(sol .- sol_, 1), 0 ,atol = 1e-1)

        @test length(Ψ) == 2
        @test size(Ψ) == (2,)
        @test parameters(Ψ) ≈ [1.0; -0.2; -9.81; -0.1]
        W = get_coefficients(Ψ)
        @test size(W) == (length(basis), 2)
        @test_nowarn get_aicc(Ψ)
        @test sum(get_sparsity(Ψ)) == 4
        @test sum(get_error(Ψ)) < 1e-10
        @test Ψ(u0) ≈ pendulum(u0, nothing, nothing)
    end

    @testset "Full Dataset" begin
        opts = [STLSQ(1e-2), SR3(1e-2, 1.0), SR3(1e-2, 1.0, ClippedAbsoluteDeviation())]

        maxiters = 50000
        for opt in opts
            Ψ = SINDy(X, DX, basis, opt, maxiter=maxiters, denoise=false, normalize=false)

            @test all(get_sparsity(Ψ) .== [1; 3])
            @test max(get_error(Ψ)...) < 1e-5
            @test all(isapprox.(parameters(Ψ), [1.0; -0.2; -9.81; -0.1],atol = 1e-1))
        end
    end

    @testset "Noisy Dataset" begin
        opts = [STLSQ(1e-1), ADMM(1e-1, 0.5), SR3(1e-1, 0.2, SoftThreshold())]
        maxiters = 50000
        λs = exp10.(-10:0.1:10)
        for opt in opts
            Ψ = SINDy(Xₙ, DX, basis, λs, opt, maxiter=maxiters, denoise=true, normalize=true)
            @test all(get_sparsity(Ψ) .== [1; 2])
            @test max(get_error(Ψ)...) < 10
        end
    end

    @testset "Limited Dataset" begin
        opts = [STLSQ(1e-2), SR3(1e-2, 0.01, SoftThreshold())]
        maxiters = 100000
        λs = exp10.(-5:0.1:5)
        for opt in opts
            Ψ = SINDy(Xₛ, DXₛ, basis, λs, opt, maxiter=maxiters, denoise=false, normalize=false)
            @test all(get_sparsity(Ψ) .== [1; 3])
            @test max(get_error(Ψ)...) < 1e-1
            @test all(isapprox.(parameters(Ψ), [1.0; -0.2; -9.81; -0.1],atol = 2e-1))
        end
    end

    @testset "Sparse Regression" begin
        opt = SR3(1e-2, 1.0)
        maxiters = 50000
        θ = basis(X)
        Ξ1, iters_1 = sparse_regression(X, DX, basis, parameters(basis), [], maxiters, opt, false, false, eps())
        Ξ2 = similar(Ξ1); Ξ3 = similar(Ξ1);
        iters_2 = sparse_regression!(Ξ2, X, DX, basis, parameters(basis), [], maxiters, opt, false, false, eps())
        iters_3 = sparse_regression!(Ξ3, θ, DX, maxiters, opt, false, false, eps())

        @test iters_1 == iters_2 == iters_3
        @test Ξ1 ≈ Ξ2 ≈ Ξ3
        @test isapprox(norm(Ξ1' * θ - DX, 2), 0; atol=1e-1)
    end
end
