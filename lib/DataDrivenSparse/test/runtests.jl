using DataDrivenDiffEq
using DataDrivenSparse
using SafeTestsets

@safetestset "Basic Sparse Regression" begin 
    include("./sparse_linear_solve.jl")
end

@safetestset "DataDrivenAPI" begin 
end