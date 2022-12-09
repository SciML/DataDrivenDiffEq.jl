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
end end
