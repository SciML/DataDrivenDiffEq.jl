using DataDrivenDiffEq
using LinearAlgebra

@testset "Evaluation" begin
    @variables u[1:3] du[1:3]

    basis = Basis(du .+ u, u, implicits = du)
    @test isequal(DataDrivenDiffEq.implicit_variables(basis), collect(du))
    x = randn(3, 10)
    dx = randn(3, 10)
    res = x .+ dx
    res_ = similar(res)
    prob = DirectDataDrivenProblem(x, dx)
    basis(res_, prob)

    @test basis(prob) == res
    @test res_ == res

    prob = DiscreteDataDrivenProblem(x)
    res = x[:, 1:(end - 1)] .+ x[:, 2:end]
    res_ = similar(res)
    basis(res_, prob)
    @test basis(prob) == res
    @test res_ == res
end

@testset "Solve implicit result" begin
    @variables u[1:3] du[1:3]
    @parameters p[1:3]
    u = collect(u)
    du = collect(du)
    p = collect(p)

    basis = Basis([du; u; sin.(p .* u)], u, parameters = p, implicits = du)
    eqs = map(x -> x.rhs, equations(basis))

    Ξ = zeros(Float32, 3, 9)
    Ξ[1, 1] = -2
    Ξ[2, 2] = 1
    Ξ[3, 3] = 5

    Ξ[1, 4] = -2
    Ξ[1, 5] = 1
    Ξ[2, 6] = 1
    Ξ[2, 4] = 3
    Ξ[3, 6] = 1

    x = randn(3, 10)
    dx = randn(3, 10)
    t = 1:1:10.0
    res = x .+ dx
    direct_prob = DirectDataDrivenProblem(x, dx)
    cont_prob = ContinuousDataDrivenProblem(x, t, DX = dx)
    discrete_prob = DiscreteDataDrivenProblem(x)

    d = Difference(get_iv(basis), dt = 1.0)
    ∂ = Differential(get_iv(basis))

    direct_res = DataDrivenDiffEq.__construct_basis(
        Ξ, basis, direct_prob,
        DataDrivenDiffEq.DataDrivenCommonOptions()
    )
    discrete_res = DataDrivenDiffEq.__construct_basis(
        Ξ, basis, discrete_prob,
        DataDrivenDiffEq.DataDrivenCommonOptions()
    )
    cont_res = DataDrivenDiffEq.__construct_basis(
        Ξ, basis, cont_prob,
        DataDrivenDiffEq.DataDrivenCommonOptions()
    )

    for r in [direct_res, discrete_res, cont_res]
        lhs = Num.(map(eq -> eq.rhs, equations(direct_res)))
        # get_variables returns Sets in Symbolics v7, so use union to combine them
        xs = collect(reduce(union, Symbolics.get_variables.(lhs); init = Set()))
        @test !any(DataDrivenDiffEq.is_dependent(Num.(xs), du))
        @test any(DataDrivenDiffEq.is_dependent(Num.(xs), u))
    end

    # Note: This is purely testing for functionality!
    basis = Basis(du .+ u, u, implicits = du)
    K = Float32[0 3 0; 2 0 1; 0 0 0.5]
    imp_basis = DataDrivenDiffEq.__construct_basis(
        K, basis, discrete_prob,
        DataDrivenDiffEq.DataDrivenCommonOptions()
    )
    @test all(isequal.(equations(imp_basis), collect(du .~ -u)))
end
