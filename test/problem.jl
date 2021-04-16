# Generate some test data
t = collect(0:0.1:5.0)
f_(t) = [sin(t); cos(t)]
df_(t) = [cos(t); -sin(t)]
X = hcat(map(i -> f_(i), t)...)
DX = hcat(map(i -> df_(i), t)...)
p = [2.7]
u_(u, p, t) = [p[1] * u[1] - 3.0 * t]
U = hcat(map(i -> u_(X[:, i], p, t[i]), 1:length(t))...)

@testset "DiscreteProblem" begin
    p1 = DiscreteDataDrivenProblem(X)
    p2 = DiscreteDataDrivenProblem(X, t)
    p3 = DiscreteDataDrivenProblem(X[:, 1:end-1], t[1:end], X[:, 2:end])
    p4 = DiscreteDataDrivenProblem(X, t, U)
    p5 = DiscreteDataDrivenProblem(X[:, 1:end-1], t[1:end], X[:, 2:end], U[:, 1:end-1])
    p6 = DiscreteDataDrivenProblem(X, t, DX, u_, p = p)

    @test is_valid(p1)
    @test is_discrete(p1)
    @test is_valid(p2)
    @test is_valid(p3)
    @test isequal(p3.X, p1.X)
    @test is_valid(p4)
    @test is_valid(p5)
    @test !is_continuous(p5)
    @test !is_valid(p6)
    @test isequal(p5.U, p6.U)
end

@testset "ContinuousProblem" begin
    p2 = ContinuousDataDrivenProblem(X, t, GaussianKernel())
    p3 = ContinuousDataDrivenProblem(X, DX)
    p4 = ContinuousDataDrivenProblem(X, t, U)
    p5 = ContinuousDataDrivenProblem(X, t, DX, U)
    p6 = ContinuousDataDrivenProblem(X, t, DX, u_, p = p)


    @test is_valid(p2)
    @test is_continuous(p2)
    @test is_valid(p3)
    @test isapprox(p3.X, p2.X, atol = 0.3)
    @test isapprox(p3.DX, p2.DX, atol = 0.3)
    @test is_valid(p4)
    @test is_valid(p5)
    @test !is_discrete(p5)
    @test is_valid(p6)
    @test isequal(p5.U, p6.U)
end
