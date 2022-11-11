using DataDrivenDiffEq
using DataDrivenDMD
using SafeTestsets
using Test

@info "Finished loading packages"

const GROUP = get(ENV, "GROUP", "All")

@time begin if GROUP == "All" || GROUP == "DataDrivenDMD"
    @safetestset "Linear autonomous" begin include("./linear_autonomous.jl") end
    @safetestset "Linear forced" begin include("./linear_forced.jl") end
    @safetestset "Nonlinear autonomous" begin include("./nonlinear_autonomous.jl") end
end end