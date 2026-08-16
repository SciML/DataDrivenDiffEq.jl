using DataDrivenDiffEq
using DataDrivenSR
using Pkg
using Test
using StableRNGs

const GROUP = get(ENV, "DATADRIVENDIFFEQ_TEST_GROUP", get(ENV, "GROUP", "All"))

function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.develop(
        [
            Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..", "..")),
            Pkg.PackageSpec(path = joinpath(@__DIR__, "..")),
        ]
    )
    return Pkg.instantiate()
end

if GROUP == "All" || GROUP == "Core" || GROUP == "DataDrivenSR"
    @testset "Developer interface" begin
        include("./Core/developer_api.jl")
    end

    @testset "Symbolic Regression" begin
        include("./Core/symbolic_regression.jl")
    end
end

if GROUP == "QA"
    activate_qa_env()
    include("qa/qa.jl")
end
