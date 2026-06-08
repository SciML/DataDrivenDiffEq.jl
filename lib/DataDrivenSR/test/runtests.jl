using DataDrivenDiffEq
using DataDrivenSR
using Test
using StableRNGs

const GROUP = get(ENV, "DATADRIVENDIFFEQ_TEST_GROUP", get(ENV, "GROUP", "All"))

if GROUP == "All" || GROUP == "Core" || GROUP == "DataDrivenSR"
    @testset "Symbolic Regression" begin
        include("./Core/symbolic_regression.jl")
    end
end
