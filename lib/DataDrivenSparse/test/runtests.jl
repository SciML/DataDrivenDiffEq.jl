using DataDrivenDiffEq
using DataDrivenSparse
using SafeTestsets

const GROUP = get(ENV, "DATADRIVENDIFFEQ_TEST_GROUP", get(ENV, "GROUP", "All"))

if GROUP == "All" || GROUP == "Core" || GROUP == "DataDrivenSparse"
    @safetestset "Basic Sparse Regression" begin
        include("./sparse_linear_solve.jl")
    end

    @safetestset "Pendulum" begin
        include("./pendulum.jl")
    end

    @safetestset "Michaelis Menten" begin
        include("./michaelis_menten.jl")
    end

    @safetestset "Cartpole" begin
        include("./cartpole.jl")
    end
end
