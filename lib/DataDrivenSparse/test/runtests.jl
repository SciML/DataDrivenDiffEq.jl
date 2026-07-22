using DataDrivenDiffEq
using DataDrivenSparse
using Pkg
using SafeTestsets

const GROUP = get(ENV, "DATADRIVENDIFFEQ_TEST_GROUP", get(ENV, "GROUP", "All"))

function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    # On Julia 1.10, QA can otherwise resolve registered copies of the in-repo
    # packages. Develop the local root and sublibrary before instantiating.
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

if GROUP == "All" || GROUP == "Core" || GROUP == "DataDrivenSparse"
    @safetestset "Basic Sparse Regression" begin
        include("./Core/sparse_linear_solve.jl")
    end

    @safetestset "Pendulum" begin
        include("./Core/pendulum.jl")
    end

    @safetestset "Michaelis Menten" begin
        include("./Core/michaelis_menten.jl")
    end

    @safetestset "Cartpole" begin
        include("./Core/cartpole.jl")
    end
end

if GROUP == "QA"
    activate_qa_env()
    @safetestset "QA" begin
        include("qa/qa.jl")
    end
end
