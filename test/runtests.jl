using DataDrivenDiffEq
using DataDrivenDiffEq.Optimize
using ModelingToolkit
using LinearAlgebra
using SafeTestsets
using Random

@info "Loading OrdinaryDiffEq"
using OrdinaryDiffEq
using Test
@info "Finished loading packages"

const GROUP = get(ENV, "GROUP", "All")

@time begin
    if GROUP == "All" || GROUP == "DataDrivenDiffEq" || GROUP == "Standard"
        @testset "Basis" begin include("./basis/basis.jl") end
        @testset "Basis Generators" begin include("./basis/generators.jl") end
        @testset "DataDrivenProblem" begin include("./problem.jl") end

        # TODO Fail right now because of scoping.
        # Should be a quick fix tomorrow
        @testset "Sparse Identification" begin
            @testset "Pendulum" begin include("./sindy/pendulum.jl") end
            @testset "Michaelis Menten" begin include("./sindy/michaelis_menten.jl") end
        end

        #include("./koopman.jl")
        #include("./isindy.jl")
        include("./utils.jl")
        #include("./optimize.jl")
    end

    # These are excluded right now, until the deps are figured out
    #if GROUP == "Integration" || GROUP == "All"
    #    @safetestset "Partial Lotka Volterra Discovery " begin include("./applications/partial_lotka_volterra.jl") end
    #end
end
