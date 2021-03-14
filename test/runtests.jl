using DataDrivenDiffEq
#using DataDrivenDiffEq.Optimize
using ModelingToolkit
using LinearAlgebra
using SafeTestsets

#@info "Loading OrdinaryDiffEq"
#using OrdinaryDiffEq
using Test
@info "Finished loading packages"

const GROUP = get(ENV, "GROUP", "All")


@time begin
    if GROUP == "All" || GROUP == "DataDrivenDiffEq" || GROUP == "Standard"
        include("./basis.jl")
        include("./problem.jl")

        #include("./koopman.jl")
        #include("./sindy.jl")
        #include("./isindy.jl")
        #include("./utils.jl")
        #include("./optimize.jl")
    end

    # These are excluded right now, until the deps are figured out
    #if GROUP == "Integration" || GROUP == "All"
    #    @safetestset "Partial Lotka Volterra Discovery " begin include("./applications/partial_lotka_volterra.jl") end
    #end
end
