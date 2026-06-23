using DataDrivenDiffEq
using DataDrivenLux
using Pkg
using SafeTestsets
using Test

@info "Finished loading packages"

const GROUP = get(ENV, "DATADRIVENDIFFEQ_TEST_GROUP", get(ENV, "GROUP", "All"))

function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    # On Julia < 1.11 the qa env's [sources] table is ignored, so the in-repo
    # DataDrivenLux/DataDrivenDiffEq would resolve as registered packages and QA
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

@testset "DataDrivenLux" begin
    if GROUP == "All" || GROUP == "Core" || GROUP == "DataDrivenLux"
        @testset "Lux" begin
            @safetestset "Nodes" include("Core/nodes.jl")
            @safetestset "Layers" include("Core/layers.jl")
            @safetestset "Graphs" include("Core/graphs.jl")
        end

        @testset "Caches" begin
            @safetestset "Candidate" include("Core/candidate.jl")
            @safetestset "Cache" include("Core/cache.jl")
        end

        @testset "Algorithms" begin
            @safetestset "RandomSearch" include("Core/randomsearch_solve.jl")
            @safetestset "Reinforce" include("Core/reinforce_solve.jl")
            @safetestset "CrossEntropy" include("Core/crossentropy_solve.jl")
        end
    end
end

if GROUP == "QA"
    activate_qa_env()
    @safetestset "QA" begin
        include("qa/qa.jl")
    end
end
