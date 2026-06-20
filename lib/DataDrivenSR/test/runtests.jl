using DataDrivenDiffEq
using DataDrivenSR
using Pkg
using Test
using StableRNGs

const GROUP = get(ENV, "DATADRIVENDIFFEQ_TEST_GROUP", get(ENV, "GROUP", "All"))

function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    # On Julia < 1.11 the qa env's [sources] table is ignored, so the in-repo
    # DataDrivenSR/DataDrivenDiffEq would resolve as registered packages and QA
    # would analyze stale released code. Develop the local paths to restore the
    # 1.11+ [sources] behavior (no-op effect on >= 1.11, which honors [sources]).
    if VERSION < v"1.11.0-DEV.0"
        Pkg.develop(
            [
                Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..", "..")),
                Pkg.PackageSpec(path = joinpath(@__DIR__, "..")),
            ]
        )
    end
    return Pkg.instantiate()
end

if GROUP == "All" || GROUP == "Core" || GROUP == "DataDrivenSR"
    @testset "Symbolic Regression" begin
        include("./Core/symbolic_regression.jl")
    end
end

if GROUP == "QA"
    activate_qa_env()
    include("qa/qa.jl")
end
