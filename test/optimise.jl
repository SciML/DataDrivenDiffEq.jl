
function generate_data(x_dim::Int64, y_dim::Int64, sparsity::Float64, ratio::Float64)
    n, m, k = x_dim, y_dim, max(floor(Int64,ratio*x_dim), y_dim)
    x = randn(n,k) # Fully random input vector
    # Generate random sparse matrix we want to recover
    A = Matrix(sprandn(m,n, sparsity))
    y = A*x # measurements
    return y, A, x
end

opts = [STRRidge(), ADMM(), SR3()]


@testset "Equal Sizes" begin
    @testset for sparsity in 0.05:0.05:0.85
        y, A, x = generate_data(3, 3,sparsity, 3.0)
        threshold = min(1/norm(y, Inf), 1e-1)^2

        @testset for opt in opts
            set_threshold!(opt, threshold)
            Ξ = DataDrivenDiffEq.Optimise.init(opt, x', y')
            fit!(Ξ, x', y', opt, maxiter = 1000)
            #@test norm(A-Ξ') < 1e-10
            @test A ≈ Ξ' atol = 1e-3
        end
    end

end

@testset "Single Signal" begin
    @testset for sparsity in 0.05:0.05:0.85
        y, A, x = generate_data(10, 1,sparsity, 3.0)
        threshold = min(1/norm(y, Inf), 1e-1)^2

        @testset for opt in opts
            set_threshold!(opt, threshold)
            Ξ = DataDrivenDiffEq.Optimise.init(opt, x', y')
            fit!(Ξ, x', y', opt, maxiter = 1000)
            #@test norm(A-Ξ') < 1e-10
            @test A ≈ Ξ' atol = 1e-3
        end
    end

end


@testset "Multiple Signals" begin
    @testset for sparsity in 0.05:0.05:0.85
        y, A, x = generate_data(50, 5,sparsity, 1.0)
        threshold = min(1/norm(y, Inf), 1e-1)^2

        @testset for opt in opts
            set_threshold!(opt, threshold)
            Ξ = DataDrivenDiffEq.Optimise.init(opt, x', y')
            fit!(Ξ, x', y', opt, maxiter = 100)
            @test norm(A-Ξ') < 1e-1
        end
    end
end

@testset "ADM" begin
    x = randn(3, 100)
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
        @test norm(θ'*L) ≈ norm(θ'*M)
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
        @test norm(θ'*L) ≈ norm(θ'*M)
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
        @test norm(θ'*L) ≈ norm(θ'*M)
        pareto = map(q->norm([norm(q, 0) ;norm(θ'*q, 2)], 2), eachcol(M))
        score, posmin = findmin(pareto)
        # Get the corresponding eqs
        q_best = M[:, posmin] ./ M[1, posmin]
        @test q_best ≈ [1.0 0 0.0 0.0 0.0 0.5 -0.5 0 -1.5]'
    end
end
