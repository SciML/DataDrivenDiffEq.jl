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

struct InterfaceBasis <: DataDrivenDiffEq.AbstractBasis
    eqs
    unknowns
    ctrls
    ps
    observed
    iv
    implicit
    f
    name
    systems
end

function (basis::InterfaceBasis)(u, p, t)
    return basis.f(u, p, t)
end

DataDrivenDiffEq.is_implicit(::InterfaceBasis) = false
DataDrivenDiffEq.is_controlled(::InterfaceBasis) = false

struct InterfaceProblem <: DataDrivenDiffEq.AbstractDataDrivenProblem{
        Float64, false, DataDrivenDiffEq.DDProbType(1),
    }
    X
    t
    DX
    Y
    U
    p
    name
end

DataDrivenDiffEq.get_oop_args(problem::InterfaceProblem) =
    (problem.X, problem.p, problem.t, problem.U)

function DataDrivenDiffEq.remake_problem(problem::InterfaceProblem; p = problem.p, kwargs...)
    return InterfaceProblem(
        problem.X, problem.t, problem.DX, problem.Y, problem.U, p, problem.name
    )
end

(basis::InterfaceBasis)(problem::InterfaceProblem) = 2 .* problem.X

struct InterfaceResult <: DataDrivenDiffEq.AbstractDataDrivenResult
    value::Float64
end

import StatsAPI
StatsAPI.coef(result::InterfaceResult) = result.value
StatsAPI.rss(result::InterfaceResult) = result.value
StatsAPI.dof(::InterfaceResult) = 1
StatsAPI.nobs(::InterfaceResult) = 1
StatsAPI.loglikelihood(::InterfaceResult) = 0.0
StatsAPI.nullloglikelihood(::InterfaceResult) = 0.0
StatsAPI.r2(::InterfaceResult) = 1.0

@testset "Generic extension interfaces" begin
    @variables x t
    f(u, p, t) = 2 .* u
    basis = InterfaceBasis(
        [x ~ 2x], [x], Any[], Any[], Any[], t, Any[], f, :interface, Any[]
    )
    X = [1.0 2.0 3.0]
    problem = InterfaceProblem(
        X, [0.0, 1.0, 2.0], zeros(0, 0), 2 .* X,
        zeros(0, 0), Float64[], :interface
    )
    algorithm = InterfaceTestAlgorithm()

    @test DataDrivenDiffEq.dynamics(basis)([3.0], [], 0.0) == [6.0]
    @test DataDrivenDiffEq.get_f(basis) === f
    @test all(isequal.(DataDrivenDiffEq.states(basis), [x]))
    @test DataDrivenDiffEq.controls(basis) == []
    @test DataDrivenDiffEq.is_direct(problem)
    @test DataDrivenDiffEq.is_autonomous(problem)
    @test !DataDrivenDiffEq.is_parametrized(problem)
    @test DataDrivenDiffEq.has_timepoints(problem)
    @test DataDrivenDiffEq.get_implicit_data(problem) == 2 .* X
    @test DataDrivenDiffEq.get_oop_args(problem) ==
        (X, Float64[], [0.0, 1.0, 2.0], zeros(0, 0))
    @test DataDrivenDiffEq.assert_lhs(problem) == (:direct, 0.0)
    @test DataDrivenDiffEq.is_valid(problem)

    inputs, targets = DataDrivenDiffEq.get_fit_targets(algorithm, problem, basis)
    @test inputs == 2 .* X
    @test targets == 2 .* X
    @test DataDrivenDiffEq.remake_problem(problem; p = [3.0]).p == [3.0]

    result = InterfaceResult(1.0)
    @test StatsAPI.coef(result) == 1.0
    @test StatsAPI.rss(result) == 1.0
    @test StatsAPI.dof(result) == 1
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
