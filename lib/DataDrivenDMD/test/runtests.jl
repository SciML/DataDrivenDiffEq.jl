using DataDrivenDiffEq
using DataDrivenDMD
using Pkg
using SafeTestsets
using Test

@info "Finished loading packages"

const GROUP = get(ENV, "DATADRIVENDIFFEQ_TEST_GROUP", get(ENV, "GROUP", "All"))

function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    # On Julia < 1.11 the qa env's [sources] table is ignored, so the in-repo
    # DataDrivenDMD/DataDrivenDiffEq would resolve as registered packages and QA
    # would analyze stale released code. Develop the local paths to restore the
    # 1.11+ [sources] behavior (no-op effect on >= 1.11, which honors [sources]).
    if VERSION < v"1.11.0-DEV.0"
        Pkg.develop(
            [
                Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..", "..")),
                Pkg.PackageSpec(path = joinpath(@__DIR__, "..")),
            ]
        )
    end
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
