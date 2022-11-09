using DataDrivenDiffEq
using DataDrivenDMD
using SafeTestsets
using Test

@info "Finished loading packages"

const GROUP = get(ENV, "GROUP", "All")

@time begin if GROUP == "All" || GROUP == "DataDrivenDMD"
    @safetestset "Linear autonomous" begin include("./linear_autonomous.jl") end
end end