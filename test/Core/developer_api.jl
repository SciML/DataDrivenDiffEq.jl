using DataDrivenDiffEq
using ModelingToolkit
using SciMLTesting
using StatsAPI: fit
using StatsBase: ZScoreTransform
using Test

struct InterfaceTestAlgorithm <: DataDrivenDiffEq.AbstractDataDrivenAlgorithm end

const DEVELOPER_API = (
    :AbstractBasis,
    :AbstractDataDrivenAlgorithm,
    :AbstractDataDrivenResult,
    :AbstractDataDrivenProblem,
    :ABSTRACT_DIRECT_PROB,
    :ABSTRACT_DISCRETE_PROB,
    :ABSTRACT_CONT_PROB,
    :InternalDataDrivenProblem,
    :get_fit_targets,
    :is_implicit,
    :is_controlled,
    :get_f,
    :get_implicit_data,
    :get_oop_args,
    :remake_problem,
    :assert_lhs,
    :apply_transform,
    :apply_transform!,
    :__construct_basis,
)

@testset "Developer API declarations" begin
    if isdefined(Base, :ispublic)
        public_names = SciMLTesting.public_api_names(DataDrivenDiffEq)
        @test all(name -> name in public_names, DEVELOPER_API)
    else
        @test all(name -> isdefined(DataDrivenDiffEq, name), DEVELOPER_API)
    end
end

@testset "Developer interface behavior" begin
    @variables x
    basis = Basis([x, x^2], [x])
    X = reshape([1.0, 2.0, 3.0], 1, :)
    Y = 2 .* X
    problem = DirectDataDrivenProblem(X, Y)

    @test !DataDrivenDiffEq.is_implicit(basis)
    @test !DataDrivenDiffEq.is_controlled(basis)
    @test DataDrivenDiffEq.get_f(basis) === dynamics(basis)
    @test problem isa DataDrivenDiffEq.ABSTRACT_DIRECT_PROB
    @test DataDrivenDiffEq.get_implicit_data(problem) == Y
    @test first(DataDrivenDiffEq.get_oop_args(problem)) == X
    @test DataDrivenDiffEq.assert_lhs(problem) == (:direct, 0.0)

    normalization_data = reshape([1.0, 3.0, 5.0], 1, :)
    transform = fit(DataNormalization(ZScoreTransform), normalization_data)
    expected = normalization_data ./ 2.0
    @test DataDrivenDiffEq.apply_transform(transform, normalization_data) == expected
    transformed = copy(normalization_data)
    @test DataDrivenDiffEq.apply_transform!(transform, transformed) === transformed
    @test transformed == expected

    inputs, targets = DataDrivenDiffEq.get_fit_targets(
        InterfaceTestAlgorithm(), problem, basis
    )
    @test inputs == basis(problem)
    @test targets == Y

    remade = DataDrivenDiffEq.remake_problem(problem; p = [3.0])
    @test parameters(remade) == [3.0]

    recovered = DataDrivenDiffEq.__construct_basis(
        reshape([2.0, 0.0], 1, 2), basis, problem,
        DataDrivenCommonOptions(generate_symbolic_parameters = false)
    )
    @test recovered(problem) == Y
end
