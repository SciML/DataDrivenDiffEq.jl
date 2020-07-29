
@info "Starting implicit SINDy tests"
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

    opt = ADM(1e-2)
    Ψ = ISInDy(X, DX, basis, opt = opt, maxiter = 10, rtol = 0.1)

    # Transform into ODE System
    sys = ODESystem(Ψ)
    dudt = ODEFunction(sys)
    ps = parameters(Ψ)

    @test all(get_error(Ψ) .< 1e-6)
    @test length(ps) == 11
    @test get_sparsity(Ψ) == [4; 3; 4]
    @test abs.(ps) ≈ abs.(Float64[-1/3 ; -1/3 ; -1.00 ; 2/3; 1.00 ;0.5 ;0.5 ; 1.0; 1.0; -1.0; 1.0])

    # Simulate
    estimator = ODEProblem(dudt, u0, tspan, ps)
    sol_ = solve(estimator, Tsit5(), saveat = 0.1)
    @test sol[:,:] ≈ sol_[:,:]

    
    # michaelis_menten
    function michaelis_menten(u, p, t)
        [0.6 - 1.5u[1]/(0.3+u[1])]
    end

    u0 = [0.5]
    tspan = (0.0, 4.0)
    problem = ODEProblem(michaelis_menten, u0, tspan)
    solution = solve(problem, Tsit5(), saveat = 0.1)

    X = solution[:,:] 
    DX = similar(X)
    for (i, xi) in enumerate(eachcol(X))
        DX[:, i] = michaelis_menten(xi, [], 0.0)
    end

    @variables u
    basis= Basis([u^i for i in 0:4], [u])
    opt = ADM(1e-1)
    f_target = WeightedSum([0.01 1.0], x->identity(x))
    Ψ = ISInDy(X, DX, basis, opt = opt, maxiter = 100, rtol = 0.9, alg = f_target)
    
    sys = ODESystem(Ψ)
    dudt = ODEFunction(sys)
    ps = parameters(Ψ)

    @test ps ≈ [-0.3; -1.0; 0.18; -0.92] atol = 1e-1
    # Simulate
    estimator = ODEProblem(dudt, u0, tspan, ps)
    sol_ = solve(estimator, Tsit5(), saveat = 0.1)

    @test isapprox(sol_[:,:], solution[:,:], atol = 1e-1) 

end
