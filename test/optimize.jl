opts = [STRRidge(), ADMM(), SR3()]
iters = Int64[3000, 30000, 30000]
atols = Float64[1e-10, 1e-4, 1e-7]

@info "Starting optimzation tests"
@testset "Equal Sizes" begin

    x = 10.0*[1 -2 3; 5 0.5 8; 1.1 2.7 5]
    A = [0.6 0 -0.1; 0.1 -8.0 0; 0.9 0 -0.8]
    y = A*x

    threshold = 1e-3

    @testset for (opt, maxiter, a_tol) in zip(opts, iters, atols)
        set_threshold!(opt, threshold)
        Ξ = DataDrivenDiffEq.Optimize.init(opt, x', y')
        _iters = fit!(Ξ, x', y', opt, maxiter = maxiter)
        @debug println("$opt $_iters $(norm(A-Ξ', 2))")
        @test _iters <= maxiter
        @test norm(A - Ξ', 2) < a_tol
    end
end

@testset "Single Signal" begin
    x = 10.0*randn(3, 100)
    A = [1.0 0 -0.1]
    y = A*x
    threshold = 1e-2
    @testset for (opt, maxiter, a_tol) in zip(opts, iters, atols)
        set_threshold!(opt, threshold)
        Ξ = DataDrivenDiffEq.Optimize.init(opt, x', y')
        _iters = fit!(Ξ, x', y', opt, maxiter = maxiter)
        @debug println("$opt $_iters $(norm(A-Ξ', 2))")
        @test _iters <= maxiter
        @test norm(A - Ξ', 2) < a_tol
    end
end

@testset "Multiple Signals" begin
    x = 10.0*randn(100, 1000)
    A = zeros(5,100)
    A[1,1] = 1.0
    A[1, 50] = 3.0
    A[2, 75] = 10.0
    A[3, 5] = -2.0
    A[4,80] = 0.2
    A[5,5] = 0.1
    y = A*x
    threshold =1e-2
    @testset for (opt, maxiter, a_tol) in zip(opts, iters, atols)
        set_threshold!(opt, threshold)
        Ξ = DataDrivenDiffEq.Optimize.init(opt, x', y')
        _iters = fit!(Ξ, x', y', opt, maxiter = maxiter)
        @debug println("$opt $_iters $(norm(A-Ξ', 2))")
        @test _iters <= maxiter
        @test norm(A - Ξ', 2) < a_tol
    end
end


@testset "ADM" begin
    x = 10.0*randn(3, 100)
    A = Float64[1 0 3; 0 1 0; 0 2 1]
    @testset "Linear" begin
        Z = A*x # Measurements
        Z[1, :] = Z[1,:] ./ (1 .+ x[2,:])
        θ = [Z[1,:]'; Z[1,:]' .* x[1,:]';Z[1,:]' .* x[2,:]';Z[1,:]' .* x[3,:]'; x[1,:]'; x[2,:]'; x[3,:]']
        M = nullspace(θ', rtol = 0.99)
        L = deepcopy(M)
        opt = ADM(1e-2)
        fit!(M, L', opt, maxiter = 10000)
        @test all(norm.(eachcol(M)) .≈ 1)
        pareto = map(q->norm([norm(q, 0) ;norm(θ'*q, 2)], 2), eachcol(M))
        score, posmin = findmin(pareto)
        # Get the corresponding eqs
        q_best = M[:, posmin] ./ M[1, posmin]
        @test q_best ≈ [1.0 0 1.0 0 -1 0 -3]'
    end

    @testset "Quadratic" begin
        Z = A*x # Measurements
        Z[1, :] = Z[1,:] ./ (1 .+ x[2,:].*x[1,:])
        θ = [Z[1,:]'; Z[1,:]' .* x[1,:]';Z[1,:]' .* x[2,:]';Z[1,:]' .* x[3,:]'; Z[1,:]' .* (x[1,:].*x[2,:])'; x[1,:]'; x[2,:]'; x[3,:]']
        M = nullspace(θ', rtol = 0.99)
        L = deepcopy(M)
        opt = ADM(1e-2)
        fit!(M, L', opt, maxiter = 10000)
        @test all(norm.(eachcol(M)) .≈ 1)
        pareto = map(q->norm([norm(q, 0) ;norm(θ'*q, 2)], 2), eachcol(M))
        score, posmin = findmin(pareto)
        # Get the corresponding eqs
        q_best = M[:, posmin] ./ M[1, posmin]
        @test q_best ≈ [1.0 0 0.0 0.0 1.0 -1 0 -3]'
    end

    @testset "Nonlinear" begin
        Z = A*x # Measurements
        Z[1, :] = Z[1,:] ./ (2 .+ sin.(x[1,:]))
        θ = [Z[1,:]'; Z[1,:]' .* x[1,:]';Z[1,:]' .* x[2,:]';Z[1,:]' .* x[3,:]'; Z[1,:]' .* (x[1,:].*x[2,:])';Z[1,:]' .* sin.(x[1,:])' ;x[1,:]'; x[2,:]'; x[3,:]']
        M = nullspace(θ', rtol = 0.99)
        L = deepcopy(M)
        opt = ADM(1e-2)
        fit!(M, L', opt, maxiter = 10000)
        @test all(norm.(eachcol(M)) .≈ 1)
        pareto = map(q->norm([norm(q, 0) ;norm(θ'*q, 2)], 2), eachcol(M))
        score, posmin = findmin(pareto)
        # Get the corresponding eqs
        q_best = M[:, posmin] ./ M[1, posmin]
        @test q_best ≈ [1.0 0 0.0 0.0 0.0 0.5 -0.5 0 -1.5]'
    end
end
