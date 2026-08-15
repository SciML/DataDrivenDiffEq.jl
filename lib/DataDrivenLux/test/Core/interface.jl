using DataDrivenDiffEq
using DataDrivenLux
using IntervalArithmetic: interval
using ModelingToolkit: @variables
using Test

struct InterfaceAlgorithm <: DataDrivenLux.AbstractDAGSRAlgorithm
    options::DataDrivenLux.CommonAlgOptions
end

DataDrivenLux.init_model(
    ::InterfaceAlgorithm, basis::Basis, dataset::DataDrivenLux.Dataset, intervals
) = DataDrivenLux.LayeredDAG(
    length(basis), size(dataset.y, 1), 1, (1,), (identity,)
)

DataDrivenLux.update_parameters!(
    ::DataDrivenLux.SearchCache{<:InterfaceAlgorithm}
) = nothing

@variables x
basis = Basis([x], [x])
problem = DirectDataDrivenProblem(
    reshape([1.0, 2.0, 3.0], 1, :), reshape([2.0, 4.0, 6.0], 1, :)
)
dataset = DataDrivenLux.Dataset(problem)
intervals = [interval(-10.0, 10.0)]
algorithm = InterfaceAlgorithm(DataDrivenLux.CommonAlgOptions())

@testset "Generic DAG symbolic-regression interface" begin
    model = DataDrivenLux.init_model(algorithm, basis, dataset, intervals)
    @test model isa DataDrivenLux.LayeredDAG

    cache = DataDrivenLux.SearchCache{
        InterfaceAlgorithm, DataDrivenLux.__PROCESSUSE(1), Nothing,
    }(
        algorithm, DataDrivenLux.Candidate[], Int[], Bool[], Int[], Float32[],
        dataset, nothing
    )
    @test DataDrivenLux.update_parameters!(cache) === nothing
end
