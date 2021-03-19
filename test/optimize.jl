@testset "Explicit Optimizer" begin
    x = 10.0*[1 -2 3; 5 0.5 8; 1.1 2.7 5]
    A = [0.6 0 -0.1; 0.1 -8.0 0; 0.9 0 -0.8]
    y = A*x
    ŷ = reshape(y[1,:], 1, 3)

    opts = [STLSQ(1e-2); ADMM(1e-2); SR3(1e-2)]

    @testset "Single Signal" begin
        for opt in opts
            Ξ = init(opt, x', ŷ')
            opt(Ξ, x', ŷ', maxiter = 100)
            @test Ξ ≈ A[1, :]
            Ξ = init(opt, x', ŷ')
            sparse_regression!(Ξ, x', ŷ', opt)
            @test Ξ ≈ A[1, :]
        end
    end
    @testset "Multi Signal" begin
        for opt in opts
            Ξ = init(opt, x', y')
            opt(Ξ, x', y', maxiter = 100)
            @test Ξ ≈ A'

            Ξ = init(opt, x', y')
            sparse_regression!(Ξ, x', y', opt)
            @test Ξ ≈ A'
        end
    end

    λs = exp10.(-3:0.1:1)
    opts = [STLSQ(λs); ADMM(λs); SR3(λs)]

    @testset "Big Sparse System" begin
        x = 10.0*randn(100, 1000)
        A = zeros(5,100)
        A[1,1] = 1.0
        A[1, 50] = 3.0
        A[2, 75] = 10.0
        A[3, 5] = -2.0
        A[4,80] = 0.2
        A[5,5] = 0.1
        y = A*x
        ŷ = reshape(y[1,:], 1, 1000)
        @testset for opt in opts
            Ξ = init(opt, x', y')
            opt(Ξ, x', y')
            @test Ξ ≈ A'
            Ξ = init(opt, x', ŷ')
            opt(Ξ, x', ŷ')
            @test Ξ ≈ A[1, :]
        end
    end
end

@testset "Implicit Optimizer" begin

    @testset "ADM Implicit" begin
        x = 10.0*randn(3, 100)
        A = Float64[0 1 0; 0 0 1; 1 0 0]
        # System
        # dx + dx*x = A*x
        Z = A*x ./ ( 1 .+ x) # Measurements
        z = reshape(Z[1,:], 1, 100)
        opt = ADM(exp10.(-3:0.1:-1))
        isa(opt, Optimize.AbstractOptimizer)
        Ξref = Float64[
            1. 1. 1. ;
            1. 0  0 ;
            0  1  0;
            0  0  1;
            0  0  0;
            0  0  1;
            1 0  0;
            0 1 0
        ]
        θ = [ones(1,size(x,2)); x]
        Ξ = init(opt, θ', Z')
        opt(Ξ,θ',Z')
        Ξ .= abs.(Ξ ./ Ξ[1,1])
        @test Ξ ≈ Ξref

        Ξ = init(opt, θ', z')
        opt(Ξ,θ',z')
        Ξ .= abs.(Ξ ./ Ξ[1,1])
        @test Ξ ≈ Ξref[:, 1]
        Ξ = init(opt, θ', z')
        sparse_regression!(Ξ,θ',z', opt)
        Ξ .= abs.(Ξ ./ Ξ[1,1])
        @test Ξ ≈ Ξref[:, 1]
    end

    @testset "ADM Explicit" begin
        x = 10.0*randn(3, 100)
        A = Float64[0 1 0; 0 0 1; 1 0 0]
        # System
        # dx + dx*x = A*x
        Z = A*x # Measurements
        z = reshape(Z[1,:], 1, 100)
        opt = ADM(exp10.(-3:0.1:-1))
        isa(opt, Optimize.AbstractOptimizer)
        Ξref = Float64[
            1. 1. 1. ;
            0 0  0 ;
            0  0  0;
            0  0  0;
            0  0  0;
            0  0  1;
            1 0  0;
            0 1 0
        ]
        θ = [ones(1,size(x,2)); x]
        Ξ = init(opt, θ', Z')
        opt(Ξ,θ',Z')
        Ξ .= abs.(Ξ ./ Ξ[1,1])
        @test Ξ ≈ Ξref

        Ξ = init(opt, θ', z')
        opt(Ξ,θ',z')
        Ξ .= abs.(Ξ ./ Ξ[1,1])
        @test Ξ ≈ Ξref[:, 1]
        Ξ = init(opt, θ', z')
        sparse_regression!(Ξ,θ',z', opt)
        Ξ .= abs.(Ξ ./ Ξ[1,1])
        @test Ξ ≈ Ξref[:, 1]
    end
end
