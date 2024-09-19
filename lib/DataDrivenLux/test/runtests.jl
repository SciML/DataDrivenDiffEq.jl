using DataDrivenDiffEq
using DataDrivenLux
using SafeTestsets
using Test

@info "Finished loading packages"

const GROUP = get(ENV, "GROUP", "All")

@time begin
    if GROUP == "All" || GROUP == "DataDrivenLux"
        @safetestset "Lux" begin
            @safetestset "Nodes" include("nodes.jl")
            @safetestset "Layers" include("layers.jl")
            @safetestset "Graphs" include("graphs.jl")
        end

        @safetestset "Caches" begin
            @safetestset "Candidate" include("candidate.jl")  # FIXME
            @safetestset "Cache" include("cache.jl")
        end

        @safetestset "Algorithms" begin
            @safetestset "RandomSearch" include("randomsearch_solve.jl")  # FIXME
            @safetestset "Reinforce" include("reinforce_solve.jl")  # FIXME
            @safetestset "CrossEntropy" include("crossentropy_solve.jl")  # FIXME
        end
    end
end
