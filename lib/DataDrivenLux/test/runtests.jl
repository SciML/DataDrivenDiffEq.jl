using DataDrivenDiffEq
using DataDrivenLux
using SafeTestsets
using Test

@info "Finished loading packages"

const GROUP = get(ENV, "GROUP", "All")

@testset "DataDrivenLux" begin
    if GROUP == "All" || GROUP == "DataDrivenLux"
        @testset "Lux" begin
            @safetestset "Nodes" include("nodes.jl")
            @safetestset "Layers" include("layers.jl")
            @safetestset "Graphs" include("graphs.jl")
        end

        @testset "Caches" begin
            @safetestset "Candidate" include("candidate.jl")
            @safetestset "Cache" include("cache.jl")
        end

        @testset "Algorithms" begin
            @safetestset "RandomSearch" include("randomsearch_solve.jl")
            @safetestset "Reinforce" include("reinforce_solve.jl")
            @safetestset "CrossEntropy" include("crossentropy_solve.jl")
        end
    end
end
