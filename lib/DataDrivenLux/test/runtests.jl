using DataDrivenDiffEq
using DataDrivenLux
using Pkg
using SafeTestsets
using Test

@info "Finished loading packages"

const GROUP = get(ENV, "DATADRIVENDIFFEQ_TEST_GROUP", get(ENV, "GROUP", "All"))

function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
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
