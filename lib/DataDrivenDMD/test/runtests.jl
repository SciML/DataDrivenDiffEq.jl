using DataDrivenDiffEq
using DataDrivenDMD
using Pkg
using SafeTestsets
using Test

@info "Finished loading packages"

const GROUP = get(ENV, "DATADRIVENDIFFEQ_TEST_GROUP", get(ENV, "GROUP", "All"))

function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    return Pkg.instantiate()
end

@time begin
    if GROUP == "All" || GROUP == "Core" || GROUP == "DataDrivenDMD"
        @safetestset "Linear autonomous" begin
            include("./Core/linear_autonomous.jl")
        end
        @safetestset "Linear forced" begin
            include("./Core/linear_forced.jl")
        end
        @safetestset "Nonlinear autonomous" begin
            include("./Core/nonlinear_autonomous.jl")
        end
        @safetestset "Nonlinear forced" begin
            include("./Core/nonlinear_forced.jl")
        end
    end

    if GROUP == "QA"
        activate_qa_env()
        @safetestset "QA" begin
            include("qa/qa.jl")
        end
    end
end
