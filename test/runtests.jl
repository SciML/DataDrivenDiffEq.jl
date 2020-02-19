using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra
using OrdinaryDiffEq
using Test

@testset "Basis" begin
    @variables u[1:3]
    @parameters w[1:2]
    h = [u[1]; u[2]; cos(w[1]*u[2]+w[2]*u[3]); u[3]+u[2]]
    h_not_unique = [1u[1]; u[1]; 1u[1]^1; h]
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

    push!(basis, 1u[3]+1u[2])
    unique!(basis)
    @test size(basis) == size(h)

    g = [1.0*u[1]; 1.0*u[3]; 1.0*u[2]]
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
end

@testset "DMD" begin
    # Create some linear data
    A = [0.9 -0.2; 0.0 0.2]
    y = [[10.; -10.]]
    for i in 1:10
        push!(y, A*y[end])
    end
    X = hcat(y...)
    @test_throws AssertionError ExactDMD(X[:, 1:end-2], dt = -1.0)
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

    @test_throws AssertionError ExtendedDMD(sol[:,:], basis, dt = -1.0)
    estimator = ExtendedDMD(sol[:,:], basis)
    @test basis == estimator.basis
    basis_2 = reduce_basis(estimator, threshold = 1e-5)
    @test size(basis_2)[1] < size(basis)[1]
    estimator_2 = ExtendedDMD(sol[:,:], basis_2)
    p1 = DiscreteProblem(dynamics(estimator), u0, tspan, [])
    s1 = solve(p1,FunctionMap())
    p2 = DiscreteProblem(dynamics(estimator_2), u0, tspan, [])
    s2 = solve(p2,FunctionMap())
    p3 = DiscreteProblem(linear_dynamics(estimator_2), estimator_2(u0), tspan, [])
    s3 = solve(p3,FunctionMap())
    @test sol[:,:] ≈ s1[:,:]
    @test sol[:,:] ≈ s2[:,:]
    @test estimator_2.basis(sol[:,:])≈ s3[:,:]
    @test eigvals(estimator_2) ≈ [-0.9; -0.3]

    # Test for nonlinear system
    function nonlinear_sys(du, u, p, t)
        du[1] = sin(u[1])
        du[2] = -0.3*u[2] -0.9*u[1]
    end

    prob = DiscreteProblem(nonlinear_sys, u0, tspan)
    sol = solve(prob,FunctionMap())
    estimator = ExtendedDMD(sol[:,:], basis)
    p4 = DiscreteProblem(dynamics(estimator), u0, tspan, [])
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
    @test isa(get_dynamics(sys), ExactDMD)
    @test sys.koopman.Ã ≈[1.5 0; 0 0.1]
    @test get_input_map(sys) ≈ [1.0; 0.0]
    @test !isstable(sys)
    @test_nowarn eigen(sys)

    # Check the solution of an unforced and forced system against each other
    dudt_ = dynamics(sys)
    prob = DiscreteProblem(dudt_, X[:, 1], (0., 10.))
    sol_unforced = solve(prob,  FunctionMap())

    dudt_ = dynamics(sys, control = (u, p, t) -> -0.5u[1])
    prob = DiscreteProblem(dudt_, X[:, 1], (0., 10.))
    sol = solve(prob, FunctionMap())

    @test all(abs.(diff(sol[1,:])) .< 1e-5)
    @test sol[2,:] ≈ sol_unforced[2,:]
end


@testset "SInDy" begin

    # Create a nonlinear pendulum
    function pendulum(u, p, t)
        x = u[2]
        y = -9.81sin(u[1]) - 0.1u[2]^3 -0.2*cos(u[1])
        return [x;y]
    end

    u0 = [0.99π; -1.0]
    tspan = (0.0, 10.0)
    prob = ODEProblem(pendulum, u0, tspan)
    sol = solve(prob, Tsit5(), saveat = 0.1)


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
    h = [cos(u[1]); sin(u[1]); u[1]*u[2]; u[1]*sin(u[2]); u[2]*cos(u[2]); polys...]

    basis = Basis(h, u)

    opt = STRRidge(1e-2)
    basis = Basis(h, u, parameters = [])
    Ψ = SInDy(sol[:,:], DX, basis, opt = opt, maxiter = 2000)
    @test_nowarn set_threshold!(opt, 0.1)
    @test size(Ψ)[1] == 2

    # Simulate
    estimator = ODEProblem(dynamics(Ψ), u0, tspan, [])
    sol_ = solve(estimator,Tsit5(), saveat = 0.1)
    @test sol[:,:] ≈ sol_[:,:]

    opt = ADMM(1e-2, 0.7)
    Ψ = SInDy(sol[:,:], DX, basis, maxiter = 5000, opt = opt)
    @test_nowarn set_threshold!(opt, 0.1)
    # Simulate
    estimator = ODEProblem(dynamics(Ψ), u0, tspan, [])
    sol_2 = solve(estimator,Tsit5(), saveat = 0.1)
    @test norm(sol[:,:] - sol_2[:,:], 2) < 2e-1
    #@test sol[:,:] ≈ sol_2[:,:]

    opt = SR3(1e-2, 1.0)
    Ψ = SInDy(sol[:,:], DX, basis, maxiter = 5000, opt = opt)
    @test_nowarn set_threshold!(opt, 0.1)

    # Simulate
    estimator = ODEProblem(dynamics(Ψ), u0, tspan, [])
    sol_3 = solve(estimator,Tsit5(), saveat = 0.1)
    @test norm(sol[:,:] - sol_3[:,:], 2) < 1e-1

    # Now use the threshold adaptation
    λs = exp10.(-5:0.1:-1)
    Ψ = SInDy(sol[:,:], DX[:, :], basis, λs,  maxiter = 20, opt = opt)
    estimator = ODEProblem(dynamics(Ψ), u0, tspan, [])
    sol_4 = solve(estimator,Tsit5(), saveat = 0.1)
    @test norm(sol[:,:] - sol_4[:,:], 2) < 1e-1

    # Check for errors
    # TODO infer the type of array and automatically push this
    @test_nowarn SInDy(sol[:,:], DX[1,:]', basis, λs, maxiter = 1, opt = opt)
end

@testset "ISInDy" begin


    # Create a test problem
    function simple(u, p, t)
        return [(2.0u[2]^2 - 3.0)/(1.0 + u[1]^2); -u[1]^2/(2.0 + u[2]^2); (1-u[2])/(1+u[3]^2)]
    end

    u0 = [2.37; 1.58; -3.10]
    tspan = (0.0, 10.0)
    prob = ODEProblem(simple, u0, tspan)
    sol = solve(prob, Tsit5(), saveat = 0.1)

    # Create the differential data
    X = sol[:,:]
    DX = similar(X)
    for (i, xi) in enumerate(eachcol(X))
        DX[:, i] = simple(xi, [], 0.0)
    end

    # Create a basis
    @variables u[1:3]
    polys = ModelingToolkit.Operation[]
    # Lots of basis functions
    for i ∈ 0:5
        if i == 0
            push!(polys, u[1]^0)
        end
        for ui in u
            if i > 0
                push!(polys, ui^i)
            end
        end
    end

    basis= Basis(polys, u)

    opt = ADM(1e-3)
    Ψ = ISInDy(X, DX, basis, opt = opt, maxiter = 100, rtol = 0.1)

    # Simulate
    estimator = ODEProblem(dynamics(Ψ), u0, tspan)
    sol_ = solve(estimator, Tsit5(), saveat = 0.1)
    @test sol[:,:] ≈ sol_[:,:]
end

@testset "Utilities" begin
    t = collect(-2:0.01:2)
    U = [cos.(t).*exp.(-t.^2) sin.(2*t)]
    S = Diagonal([2.; 3.])
    V = [sin.(t).*exp.(-t) cos.(t)]
    A = U*S*V'
    σ = 0.5
    Â = A + σ*randn(401, 401)
    n_1 = norm(A-Â)
    B = optimal_shrinkage(Â)
    optimal_shrinkage!(Â)
    @test norm(A-Â) < n_1
    @test norm(A-B) == norm(A-Â)

    X = randn(3, 100)
    Y = randn(3, 100)
    k = 3

    @test AIC(k, X, Y) == 2*k-2*log(sum(abs2, X- Y))
    @test AICC(k, X, Y) == AIC(k, X, Y)+ 2*(k+1)*(k+2)/(size(X)[2]-k-2)
    @test BIC(k, X, Y) == -2*log(sum(abs2, X -Y)) + k*log(size(X)[2])
    @test AICC(k, X, Y, likelyhood = (X,Y)->sum(abs, X-Y)) == AIC(k, X, Y, likelyhood = (X,Y)->sum(abs, X-Y))+ 2*(k+1)*(k+2)/(size(X)[2]-k-2)


    # Sampling
    X = randn(Float64, 2, 100)
    t = collect(0:0.1:9.99)
    Y = randn(size(X))
    xt = burst_sampling(X, 5, 10)
    @test 10 <= size(xt)[end] <= 60
    @test all([any(xi .≈ X) for xi in eachcol(xt)])
    xt, tt = burst_sampling(X, t, 5, 10)
    @test all(diff(tt) .> 0.0)
    @test size(xt)[end] == size(tt)[end]
    @test all([any(xi .≈ X) for xi in eachcol(xt)])
    @test !all([any(xi .≈ Y) for xi in eachcol(xt)])
    xs, ts = burst_sampling(X, t, 2.0, 1)
    @test all([any(xi .≈ X) for xi in eachcol(xs)])
    @test size(xs)[end] == size(ts)[end]
    @test ts[end]-ts[1] ≈ 2.0
    X2n = subsample(X, 2)
    t2n = subsample(t, 2)
    @test size(X2n)[end] == size(t2n)[end]
    @test size(X2n)[end] == Int(round(size(X)[end]/2))
    @test X2n[:, 1] == X[:, 1]
    @test X2n[:, end] == X[:, end-1]
    @test all([any(xi .≈ X) for xi in eachcol(X2n)])
    xs, ts = subsample(X, t, 0.5)
    @test size(xs)[end] == size(ts)[end]
    @test size(xs)[1] == size(X)[1]
    @test all(diff(ts) .≈ 0.5)
    # Loop this a few times to be sure its right
    @test_nowarn for i in 1:20
        xs, ts = burst_sampling(X, t, 2.0, 1)
        xs, ts = subsample(X, t, 0.5)
    end
end
