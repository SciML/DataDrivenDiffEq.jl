using DataDrivenDiffEq
using DataDrivenLux
using SafeTestsets
using Test

@info "Finished loading packages"

const GROUP = get(ENV, "GROUP", "All")

@time begin if GROUP == "All" || GROUP == "DataDrivenLux"
    @safetestset "Lux" begin 
        @safetestset "Nodes" begin include("./nodes.jl") end
        @safetestset "Layers" begin include("./layers.jl") end
        @safetestset "Graphs" begin include("./graphs.jl") end
    end

    @safetestset "Caches" begin
        @safetestset "Candidate" begin include("./candidate.jl") end
        @safetestset "Cache" begin include("./cache.jl") end
    end
    @safetestset "Algorithms" begin
        @safetestset "RandomSearch" begin include("./randomsearch_solve.jl") end
        @safetestset "Reinforce" begin include("./reinforce_solve.jl") end
    end
end end
