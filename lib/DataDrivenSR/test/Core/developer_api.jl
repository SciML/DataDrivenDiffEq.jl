using DataDrivenDiffEq
using DataDrivenSR
using Symbolics: @variables
using Test

@testset "AbstractDataDrivenAlgorithm" begin
    @variables x
    basis = Basis([x], [x])
    X = reshape([1.0, 2.0, 3.0], 1, :)
    Y = 2 .* X
    problem = DirectDataDrivenProblem(X, Y)
    algorithm = EQSearch()

    @test algorithm isa DataDrivenDiffEq.AbstractDataDrivenAlgorithm
    inputs, targets = DataDrivenDiffEq.get_fit_targets(algorithm, problem, basis)
    @test inputs == X
    @test targets == Y
end
