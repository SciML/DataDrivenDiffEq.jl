
@info "Starting implicit SINDy tests"
@testset "ISInDy" begin
    
    @info "Nonlinear Implicit System"
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

    opt = ADM(1e-2)
    Ψ = ISInDy(X, DX, basis, opt = opt, maxiter = 100)

    # Transform into ODE System
    sys = ODESystem(Ψ)
    dudt = ODEFunction(sys)
    ps = parameters(Ψ)
    print_equations(Ψ, show_parameter = true)

    @test all(get_error(Ψ) .< 1e-6)
    @test length(ps) == 11
    @test get_sparsity(Ψ) == [4; 3; 4]
    
    # Simulate
    estimator = ODEProblem(dudt, u0, tspan, ps)
    sol_ = solve(estimator, Tsit5(), saveat = 0.1)
    @test sol[:,:] ≈ sol_[:,:]
    @test abs.(ps) ≈ abs.(Float64[-1/3 ; -1/3 ; -1.00 ; 2/3; 1.00 ;0.5 ;0.5 ; 1.0; 1.0; -1.0; 1.0])


    @info "Michaelis-Menten-Kinetics"
    # michaelis_menten
    function michaelis_menten(u, p, t)
        [0.6 - 1.5u[1]/(0.3+u[1])]
    end

    u0 = [0.5]
    tspan = (0.0, 6.0)
    
    problem_1 = ODEProblem(michaelis_menten, u0, tspan)
    solution_1 = solve(problem_1, Tsit5(), saveat = 0.1)

    problem_2 = ODEProblem(michaelis_menten, 2*u0, tspan)
    solution_2 = solve(problem_2, Tsit5(), saveat = 0.1)

    X = Array(solution_1)#[Array(solution_1) Array(solution_2)]

    DX = similar(X)
    for (i, xi) in enumerate(eachcol(X))
        DX[:, i] = michaelis_menten(xi, [], 0.0)
    end

    @variables u
    basis= Basis([u^i for i in 0:4], [u])
    Ψ = ISInDy(X, DX, basis, ADM(1.1e-1), maxiter = 100)
    print_equations(Ψ, show_parameter = true)
    sys = ODESystem(Ψ)
    dudt = ODEFunction(sys)
    ps = parameters(Ψ)
    # Simulate
    estimator = ODEProblem(dudt, u0, tspan, ps)
    sol_ = solve(estimator, Tsit5(), saveat = 0.1)
    @test isapprox(sol_[:,:], solution_1[:,:], atol = 1e-1)
    @test abs.(ps) ≈ [1/3; 1.0; 0.2; 0.92] atol = 1e-1

    Ψ = ISInDy(X, DX, basis, STRRidge(1e-2), maxiter = 100, normalize = true)
    print_equations(Ψ, show_parameter = true)
    sys = ODESystem(Ψ)
    dudt = ODEFunction(sys)
    ps = parameters(Ψ)
    # Simulate
    estimator = ODEProblem(dudt, u0, tspan, ps)
    sol_ = solve(estimator, Tsit5(), saveat = 0.1)
    @test isapprox(sol_[:,:], solution_1[:,:], atol = 1e-1)
    @test abs.(ps) ≈ [1/3; 1.0; 0.2; 0.92] atol = 1e-1

    λs = exp10.(-3:0.1:-1)
    Ψ = ISInDy(X, DX, basis, λs ,STRRidge(1e-2), maxiter = 100, normalize = false)
    print_equations(Ψ, show_parameter = true)
    sys = ODESystem(Ψ)
    dudt = ODEFunction(sys)
    ps = parameters(Ψ)
    # Simulate
    estimator = ODEProblem(dudt, u0, tspan, ps)
    sol_ = solve(estimator, Tsit5(), saveat = 0.1)
    @test isapprox(sol_[:,:], solution_1[:,:], atol = 1e-1)
    @test abs.(ps) ≈ [1/3; 1.0; 0.2; 0.92] atol = 1e-1
end
