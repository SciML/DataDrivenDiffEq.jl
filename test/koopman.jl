algorithms = [DMDPINV(), DMDSVD(1e-2), TOTALDMD(1e-2, DMDPINV()), TOTALDMD(1e-2,DMDSVD(1e-2))]

@info "Starting Dynamic Mode Decomposition tests"
@testset "Dynamic Mode Decomposition" begin

    @info "Starting linear discrete system tests"
    @testset "Linear Discrete System" begin
        # Create some linear data
        A = [0.9 -0.2; 0.0 0.2]
        y = [[10.; -10.]]
        for i in 1:10
            push!(y, A*y[end])
        end
        X = hcat(y...)
        for alg_ in algorithms
            @info "Testing $alg_"
            estimator = DMD(X, alg = alg_)
            @test operator(estimator) ≈ A
            @test isstable(estimator)
            @test eigvals(estimator) ≈ eigvals(A)
            @test eigvecs(estimator) ≈ eigvecs(A)
            @test is_discrete(estimator)
            @test_nowarn update!(estimator, X[:, end-1], X[:,end])
            @test_throws AssertionError outputmap(estimator)
        end
    end

    @info "Starting linear continuous system tests"
    @testset "Linear Continuous System" begin
        A = [-0.9 0.1; 0.0 -0.2]
        f(u, p, t) = A*u
        u0 = [10.0; -20.0]
        prob = ODEProblem(f, u0, (0.0, 10.0))
        sol = solve(prob, Tsit5(), saveat = 0.001)
        X = sol[:,:]
        DX = sol(sol.t, Val{1})[:,:]

        for alg_ in algorithms
            @info "Testing $alg_"
            estimator = gDMD(X, DX, alg = alg_)
            @test isapprox(generator(estimator), A, atol = 1e-3)
            @test isstable(estimator)
            @test isapprox(eigvals(estimator), eigvals(A), atol = 1e-3)
            @test isapprox(abs.(eigvecs(estimator)), abs.(eigvecs(A)), atol = 1e-3)
        end
    end

    @info "Starting big continuous system tests"
    @testset "Big System" begin
        dd(u, p, t) = -0.9*ones(length(u))*u[1] - 0.05*u
        u0 = randn(100)*100.0
        u0[1] = 100.0
        prob = ODEProblem(dd, u0, (0.0, 10.0))
        sol = solve(prob, Tsit5(), saveat = 0.001)
        X = sol[:,:]
        DX = sol(sol.t, Val{1})[:,:]

        for alg_ in algorithms
            @info "Testing $alg_"
            d = gDMD(X, DX, alg = alg_)
            d2 = gDMD(sol.t, X, alg = alg_)
            d3 = gDMD(sol.t, X, alg = alg_, dt = 0.01)
            test = ODEProblem(d, u0, (0.0, 10.0))
            sol_ = solve(test, Tsit5(), saveat = 0.001)

            @test norm(sol-sol_, Inf) < 2.0
            @test isapprox(generator(d), generator(d2), atol = 1e-1)
            @test isapprox(generator(d), generator(d3), atol = 5e-1)
        end
    end

    @info "Starting low rank linear system tests"
    @testset "Rank Reduction" begin
        K = -0.5*I + [0 0 -0.2; 0.1 0 -0.1; 0. -0.2 0]
        F = qr(randn(20, 3))
        Q = F.Q[:, 1:3]
        dudt(u, p, t) = K*u
        prob = ODEProblem(dudt, [10.0; 0.3; -5.0], (0.0, 10.0))
        sol_ = solve(prob, Tsit5(), saveat = 0.01)

        # True Rank is 3
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
end


@info "Starting Extended Dynamic Mode Decomposition tests"
@testset "Extended Dynamic Mode Decomposition" begin

    @info "Starting linear discrete system tests"
    @testset "Linear Discrete System" begin
        # Test for linear system
        function linear_sys(u, p, t)
            x = -0.9 * u[1]
            y = -0.3 * u[2]
            return [x; y]
        end

        u0 = [π; 1.0]
        tspan = (0.0, 50.0)
        prob = DiscreteProblem(linear_sys, u0, tspan)
        sol = solve(prob, FunctionMap())

        @variables u[1:2]
        h = [u[1]; u[2]; sin(u[1]); cos(u[1]); u[1] * u[2]; u[2]^2]
        basis = Basis(h, u)

        estimator = EDMD(sol[:, :], basis, alg = DMDSVD())
        p1 = DiscreteProblem(estimator, u0, tspan)
        s1 = solve(p1,FunctionMap())
        basis_2 = reduce_basis(estimator, threshold = 1e-5)
        @test basis == estimator.basis
        @test size(basis_2)[1] < size(basis)[1]
        @test sol[:,:] ≈ s1[:,:]

        # Check EDMD
        for alg_ in algorithms
            @info "Testing $alg_"
            estimator_ = EDMD(sol[:,:], basis_2)
            p_ = DiscreteProblem(estimator_, u0, tspan)
            s_ = solve(p_,FunctionMap())
            @test sol[:,:] ≈ s_[:,:]
            @test eigvals(estimator_) ≈ [-0.9; -0.3]
            @test isstable(estimator_)
        end
    end

    @info "Starting linear continuous system tests"
    @testset "Linear Continuous System" begin

        function nonlinear_sys2(du, u, p, t)
            du[1] = u[2]
            du[2] = -0.9u[1]
        end

        u0 = [10.0; -20.0]

        prob_nl = ODEProblem(nonlinear_sys2, u0, (0.0, 10.0))
        sol_nl = solve(prob_nl, Tsit5())

        X = Array(sol_nl)
        DX = sol_nl(sol_nl.t, Val{1})[:,:]

        @variables u[1:2]
        basis = Basis(u, u)

        for alg_ in algorithms
            @info "Testing $alg_"
            estimator = gEDMD(X, DX, basis, alg = alg_)
            #estimator_derivative = gEDMD(sol_nl.t, X, basis, alg = alg_)
            #estimator_interpolation = gEDMD(sol_nl.t, X, basis, dt = 0.1, alg = alg_)
            @test generator(estimator) ≈ [0  1.0; -0.9 0]
            @test outputmap(estimator) ≈ I(2)
            #@test generator(estimator) ≈ generator(estimator_derivative) atol = 1e-1
            #@test generator(estimator) ≈ generator(estimator_interpolation) atol = 1e-1
        end
    end

    @info "Starting nonlinear discrete system tests"
    @testset "Nonlinear Discrete System" begin
        function nonlinear_sys(du, u, p, t)
            du[1] = 0.9u[1] + 0.1u[2]^2
            du[2] = sin(u[1]) - 0.1u[1]
        end

        u0 = [π; 1.0]
        tspan = (0.0, 50.0)

        @variables u[1:2]
        h = [u[1]; u[2]; sin(u[1]); u[2]^2]
        basis = Basis(h, u)

        prob = DiscreteProblem(nonlinear_sys, u0, tspan)
        sol = solve(prob,FunctionMap())
        estimator = EDMD(sol[:,:], basis, alg = DMDPINV())
        sys = ODESystem(estimator)
        @test isa(sys, ODESystem)
        dudt = ODEFunction(sys)
        p_ = DiscreteProblem(dudt, u0, tspan)
        s_ = solve(p_,FunctionMap())
        @test sol[:,:] ≈ s_[:,:]

        for alg_ in algorithms
            @info "Testing $alg_"
            estimator_ = EDMD(sol[:,:], basis, alg = alg_)
            p_ = DiscreteProblem(estimator_, u0, tspan)
            s_ = solve(p_,FunctionMap())
            @test sol[:,:] ≈ s_[:,:] atol = 1e-1
        end
    end

    @info "Starting nonlinear continuous system tests"
    @testset "Nonlinear Continuous System" begin

            function slow_manifold(du, u, p, t)
                du[1] = p[1]*u[1]
                du[2] = p[2]*(u[2]-u[1]^2)
            end

            u0 = [3.0; -2.0]
            tspan = (0.0, 10.0)
            p = [-0.05, -0.9]

            prob = ODEProblem(slow_manifold, u0, tspan, p)
            sol = solve(prob, Tsit5(), saveat = 0.2)

            X = Array(sol)
            # This enforces more accurate results
            DX = similar(X)
            for (i,dx) in enumerate(eachcol(DX))
                slow_manifold(dx, X[:, i], p, sol.t[i])
            end

            @variables u[1:2]
            basis = Basis([u; u[1]^2], u)

            for alg_ in algorithms
                @info "Testing $alg_"
                estimator2 = gEDMD(X, DX, basis, alg = alg_)
                A_analytical = [p[1] 0 0; 0 p[2] -p[2]; 0 0 2*p[1]]
                outputmap(estimator2)
                generator(estimator2)
                @test generator(estimator2) ≈ [p[1] 0 0; 0 p[2] -p[2]; 0 0 2*p[1]] atol =  1e-3
                @test outputmap(estimator2) ≈ [1 0 0 ; 0 1 0] atol = 1e-3
                @test abs.(modes(estimator2)) ≈ abs.(eigvecs(A_analytical)) atol = 1e-3
                @test frequencies(estimator2) ≈ eigvals(A_analytical) atol = 1e-3
            end

    end
end

@info "Starting Dynamic Mode Decomposition with Control tests"
@testset "Dynamic Mode Decomposition with Control" begin

    # Define measurements from unstable system with known control input
    X = [4 2 1 0.5 0.25; 7 0.7 0.07 0.007 0.0007]
    U = [-4 -2 -1 -0.5]
    B = Float32[1; 0]

    # But with a little more knowledge
    for alg_ in algorithms
        @info "Testing $alg_"
        sys = DMDc(X, U, B = B, alg = alg_)
        @test operator(sys) ≈[1.5 0; 0 0.1]
        @test inputmap(sys) ≈ [1.0; 0.0]
        @test !isstable(sys)
        @test_nowarn eigen(sys)
    end


    sys = DMDc(X, U, B = B)
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

    for alg_ in algorithms
        @info "Testing $alg_"
        sys4 = DMDc(X, U, alg = alg_)
        @test operator(sys4) ≈ A
        @test inputmap(sys4) ≈ B
        sys5 = gDMDc(X[:, 1:end-1], X[:, 2:end], U[:, 1:end])
        @test_throws AssertionError operator(sys5)
        @test generator(sys5) ≈ A
        @test inputmap(sys5) ≈ B
    end

    for alg_ in algorithms
        @info "Testing $alg_"
        sys6 = gDMDc(collect(0.0:9.0), X[:, 1:end-1], U, B = B, alg = alg_)
        @test exp(generator(sys6)) ≈ A atol = 1e-1
        @test inputmap(sys6) ≈ B
    end

end
