using DataDrivenDiffEq
using SafeTestsets
using Test

@info "Finished loading packages"

const GROUP = get(ENV, "GROUP", "All")

@safetestset "Basis" begin include("./basis/basis.jl") end
@safetestset "Implicit Basis" begin include("./basis/implicit_basis.jl") end
@safetestset "Basis generators" begin include("./basis/generators.jl") end
@safetestset "DataDrivenProblem" begin include("./problem/problem.jl") end
@safetestset "DataDrivenProblem Sampler" begin include("./problem/samplers.jl") end
@safetestset "DataDrivenSolution" begin include("./solution/solution.jl") end
@safetestset "Utilities" begin include("./utils.jl") end
