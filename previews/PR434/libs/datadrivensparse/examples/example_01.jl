using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra
using DataDrivenSparse

f(u) = u .^ 2 .+ 2.0u .- 1.0
X = randn(1, 100);
Y = reduce(hcat, map(f, eachcol(X)));

problem = DirectDataDrivenProblem(X, Y, name = :Test)

@variables u
basis = Basis(monomial_basis([u], 2), [u])
println(basis) # hide

res = solve(problem, basis, STLSQ())
println(res) # hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

