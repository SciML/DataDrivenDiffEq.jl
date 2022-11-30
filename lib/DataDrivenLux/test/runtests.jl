using DataDrivenDiffEq
using DataDrivenLux
using SafeTestsets
using Test

@info "Finished loading packages"

const GROUP = get(ENV, "GROUP", "All")

@time begin if GROUP == "All" || GROUP == "DataDrivenLux"
    @safetestset "Nodes and Layers" begin include("./layers_nodes.jl") end
    @safetestset "Candidates" begin include("./candidate.jl") end
    @safetestset "Cache" begin end
    @safetestset "Example" begin end
end end
