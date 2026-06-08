using DataDrivenDiffEq, Test
using LinearAlgebra
using ModelingToolkit
using StatsBase

# Generate some test data
t = collect(0:0.1:5.0)
f_(t) = [sin(t); cos(t)]
df_(t) = [cos(t); -sin(t)]
X = hcat(map(i -> f_(i), t)...)
Y = randn(2, 2) * X
DX = hcat(map(i -> df_(i), t)...)
p = [2.7]
u_(u, p, t) = [p[1] * u[1] - 3.0 * t]
U = hcat(map(i -> u_(X[:, i], p, t[i]), 1:length(t))...)

@testset "DiscreteProblem" begin
    p1 = DiscreteDataDrivenProblem(X)
    p2 = DiscreteDataDrivenProblem(X, t)
    p3 = DiscreteDataDrivenProblem(X, t, U)
    p4 = DiscreteDataDrivenProblem(X, t, u_, p = p)

    for p_ in [p1; p2; p3; p4]
        @test is_valid(p_)
        @test is_discrete(p_)
        @test_nowarn @is_applicable p_
    end

    @test isequal(p3.U, p4.U)
    @test isequal(p4.p, p)
end

@testset "ContinuousProblem" begin
    p2 = ContinuousDataDrivenProblem(X, t, GaussianKernel())
    p3 = ContinuousDataDrivenProblem(X, DX)
    p4 = ContinuousDataDrivenProblem(X, t, U)
    p5 = ContinuousDataDrivenProblem(X, t, DX, U)
    p6 = ContinuousDataDrivenProblem(X, t, DX, u_, p = p)

    for p_ in [p2; p3; p4; p5; p6]
        @test is_valid(p_)
        @test is_continuous(p_)
    end

    @test isapprox(p3.X, p2.X, atol = 0.3)
    @test isapprox(p3.DX, p2.DX, atol = 0.3)
    @test isequal(p5.U, p6.U)
    @test is_parametrized(p6)
    @test isequal(p6.p, p)
    @test has_timepoints(p4)
end

@testset "DirectProblem" begin
    p1 = DirectDataDrivenProblem(X, Y)
    p2 = DirectDataDrivenProblem(X, t, Y)
    p3 = DirectDataDrivenProblem(X, t, Y, U)
    p4 = DirectDataDrivenProblem(X, t, Y, u_, p = p)

    for p_ in [p1; p2; p3; p4]
        @test is_valid(p_)
        @test is_direct(p_)
    end

    @test isequal(p3.U, p4.U)
    @test isequal(p4.p, p)
    @test !is_autonomous(p3)
end

@testset "Problem Basis Interaction" begin
    @variables x y z t α β u
    b1 = Basis(
        [α * x; β * y; z * t^2 + u], [x; y; z], iv = t, parameters = [α; β],
        controls = [u]
    )
    b2 = Basis([α * x; β * y; z * t; α], [x; y; z], iv = t, parameters = [α; β])
    sample_size = 100
    X1 = randn(3, sample_size)
    DX1 = randn(3, sample_size)
    DX2 = randn(3, sample_size - 1)
    X2 = randn(4, sample_size)
    Y1 = randn(3, sample_size)
    Y2 = randn(4, sample_size)
    t = randn(sample_size)
    U = randn(1, sample_size)
    ps = randn(3)

    p1 = DirectDataDrivenProblem(X1, Y1, t = t, p = ps, U = U)
    p2 = DirectDataDrivenProblem(X1, Y2, t = t, p = ps, U = U)
    p3 = ContinuousDataDrivenProblem(X1, t, DX1, Y = Y1, p = ps, U = U)
    p4 = DiscreteDataDrivenProblem(X1, Y = Y2, t = t, p = ps, U = U)

    @testset "Check validity" begin
        @test_throws AssertionError @is_applicable p2 b2
        @test_throws AssertionError @is_applicable p1 b2
        @test_throws AssertionError @is_applicable p3 b2
        @test_throws AssertionError @is_applicable p4 b2
        @test_throws AssertionError @is_applicable p4 b1 DX1

        @test_nowarn @is_applicable p1 b1
        @test_nowarn @is_applicable p3 b1
        @test_nowarn @is_applicable p4 b1
        @test_nowarn @is_applicable p1 b1 DX1
        @test_nowarn @is_applicable p3 b1 DX1
    end

    @testset "Apply problem" begin
        DX1 .= 0.0
        @test_nowarn b1(DX1, p1)
        @test iszero(norm(DX1 - b1(p1)))

        DX1 .= 0.0
        @test_nowarn b1(DX1, p3)
        @test iszero(norm(DX1 - b1(p3)))

        DX2 .= 0.0
        @test_nowarn b1(DX2, p4)
        @test iszero(norm(DX2 - b1(p4)))

        DX1 .= 0.0
        @test_nowarn @views b1(DX1[:, 1:(end - 1)], p4)
        @test iszero(norm(DX1[:, 1:(end - 1)] - b1(p4)))
    end
end

@testset "DataDrivenDataset" begin
    p1 = ContinuousDataDrivenProblem(X, t)
    p2 = ContinuousDataDrivenProblem(X, t, DX = DX)
    p3 = ContinuousDataDrivenProblem(X, t, DX = DX)

    data = (
        prob1 = (X = X, t = t, Y = Y),
        prob2 = (X = X, t = t, Y = Y, DX = DX),
    )

    s1 = DataDrivenDataset(p1, p2)
    s2 = ContinuousDataset(data)
    s3 = DirectDataset(data)
    s4 = DiscreteDataset(data)
    s5 = DataDrivenDataset(p1, p2, p3)

    sets = [s1, s2, s3, s4]

    # Information
    @test is_discrete(s4)
    @test is_continuous(s2)
    @test is_direct(s3)

    # Sizes
    for s in sets
        @test size(s) ==
            (first(size(p1)), is_discrete(s) ? 2 * size(X, 2) - 2 : 2 * size(X, 2))
        @test DataDrivenDiffEq.is_valid(s)
    end

    # Basis handling
    @variables x[1:size(X, 1)]
    b = Basis(x, x)
    @test b(s1) == hcat(b(p1), b(p2))
    @test b(s2) == b(s1)
    @test hcat(X, X) == b(s3)
    @test hcat(X[:, 1:(end - 1)], X[:, 1:(end - 1)]) == b(s4)
    @test hcat(X, X, X) == b(s5)

    # Check if misspecified data is detected
    wrong_data = (
        prob1 = (X = X, Y = Y),
        prob2 = (X = X, t = t, Y = Y),
    )
    @test_throws ArgumentError ContinuousDataset(wrong_data)
end

@testset "DataDrivenProblem from ODEProblem solution" begin
    using OrdinaryDiffEqTsit5
    using ModelingToolkit: t_nounits as time, D_nounits as D

    # Define autoregulation system without @mtkmodel macro
    # (avoids macro import issues with SafeTestsets)
    @parameters α = 1.0 β = 1.3 γ = 2.0 δ = 0.5
    @variables (x(time))[1:2] = [20.0, 12.0]
    x = collect(x)

    eqs = [
        D(x[1]) ~ α / (1 + x[2]) - β * x[1],
        D(x[2]) ~ γ / (1 + x[1]) - δ * x[2],
    ]

    @named sys = System(eqs, time)
    sys_compiled = mtkcompile(sys)

    tspan = (0.0, 5.0)
    de_problem = ODEProblem{true, SciMLBase.NoSpecialize}(sys_compiled, [], tspan)
    de_solution = solve(de_problem, Tsit5(), saveat = 0.005)
    prob = DataDrivenProblem(de_solution)
    @test is_valid(prob)
end
