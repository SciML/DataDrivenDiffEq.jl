# Generate some test data
t = collect(0:0.1:5.0)
f_(t) = [sin(t); cos(t)]
df_(t) = [cos(t); -sin(t)]
X = hcat(map(i -> f_(i), t)...)
Y = randn(2,2)*X
DX = hcat(map(i -> df_(i), t)...)
p = [2.7]
u_(u, p, t) = [p[1] * u[1] - 3.0 * t]
U = hcat(map(i -> u_(X[:, i], p, t[i]), 1:length(t))...)

@testset "DiscreteProblem" begin
    p1 = DiscreteDataDrivenProblem(X)
    p2 = DiscreteDataDrivenProblem(X, t)
    p3 = DiscreteDataDrivenProblem(X, t, U)
    p4 = DiscreteDataDrivenProblem(X, t, u_, p = p)

    for p_ in [p1;p2;p3;p4]
        @test is_valid(p_)
        @test is_discrete(p_)
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

    for p_ in [p2;p3;p4;p5;p6]
        @test is_valid(p_)
        @test is_continuous(p_)
    end

    @test isapprox(p3.X, p2.X, atol = 0.3)
    @test isapprox(p3.DX, p2.DX, atol = 0.3)
    @test isequal(p5.U, p6.U)
    @test isequal(p6.p, p)
end

@testset "DirectProblem" begin
    p1 = DirectDataDrivenProblem(X, Y)
    p2 = DirectDataDrivenProblem(X, t, Y)
    p3 = DirectDataDrivenProblem(X, t, Y, U)
    p4 = DirectDataDrivenProblem(X, t, Y, u_, p = p)

    for p_ in [p1;p2;p3;p4]
        @test is_valid(p_)
        @test is_direct(p_)
    end

    @test isequal(p3.U, p4.U)
    @test isequal(p4.p, p)
end
