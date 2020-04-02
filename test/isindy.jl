
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
