
@info "Starting symbolic regression tests"
@testset "Operations and Candidates" begin
    op = OperationPool([+, *, sin, cos, tanh, exp])
    @test !all(DataDrivenDiffEq.is_unary(op, 1:2))
    @test all(DataDrivenDiffEq.is_unary(op, 3:6))
    @test length(op) == 6
    @test size(op) == (6,)
    @test DataDrivenDiffEq.random_operation(op)[1] ∈ op.ops
    @test length(DataDrivenDiffEq.random_operation(op)) == 2

    @variables u[1:2] t
    @parameters p[1:2]

    b = Basis(p.*u, u, parameters = p, iv = t)
    c = DataDrivenDiffEq.Candidate(b)
    @test all(isequal.(variables(c), variables(b)))
    @test all(isequal.(parameters(c), parameters(b)))
    @test all(isequal.(independent_variable(c), independent_variable(b)))
    @test length(c) == length(b)
    @test size(c) == (2,)
    X = randn(2, 100)
    Y1 = similar(X)
    Y2 = similar(X)
    p_ = randn(2)
    t_ = collect(1:100)
    b(Y1, X, p_, t_)
    c(Y2, X, p_, t_)
    @test Y1 == Y2
    DataDrivenDiffEq.add_features!(c, op, 2)
    @test length(c) == 4
    @test b(X, p_, t_) == c(X, p_, t_)
    DataDrivenDiffEq.add_features!(c, op, 2, 1:2, 3:4)
    @test length(c) == 4
end
