using DataDrivenDiffEq
using DataDrivenLux
using Symbolics: @variables
using Test

struct InterfaceTestAlgorithm <: DataDrivenLux.AbstractDAGSRAlgorithm end

@testset "AbstractDAGSRAlgorithm" begin
    @variables x
    basis = Basis([x], [x])
    X = reshape([1.0, 2.0, 3.0], 1, :)
    Y = 2 .* X
    problem = DirectDataDrivenProblem(X, Y)

    @test InterfaceTestAlgorithm() isa DataDrivenLux.AbstractDAGSRAlgorithm
    inputs, targets = DataDrivenDiffEq.get_fit_targets(
        InterfaceTestAlgorithm(), problem, basis
    )
    @test inputs == X
    @test targets == Y
end

@testset "Component interfaces" begin
    @test Softmax() isa DataDrivenLux.AbstractSimplex
    @test AdditiveError() isa DataDrivenLux.AbstractErrorModel
    @test RelativeReward() isa DataDrivenLux.AbstractRewardScale
end
