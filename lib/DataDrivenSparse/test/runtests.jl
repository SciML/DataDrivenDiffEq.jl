using DataDrivenDiffEq
using DataDrivenSparse
using SafeTestsets

@safetestset "Basic Sparse Regression" begin include("./sparse_linear_solve.jl") end

@safetestset "Pendulum" begin include("./pendulum.jl") end

@safetestset "Michaelis Menten" begin include("./michaelis_menten.jl") end

@safetestset "Cartpole" begin include("./cartpole.jl") end
