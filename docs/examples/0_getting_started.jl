# # [Getting Started](@id getting_started)
#
# The workflow for [DataDrivenDiffEq.jl](https://github.com/SciML/DataDrivenDiffEq.jl) is similar to other [SciML](https://sciml.ai/) packages. 
# You start by defining a [`DataDrivenProblem`](@ref) and then dispatch on the [`solve`](@ref) command to return a [`DataDrivenSolution`](@ref).

# Here is an outline of the required elements and choices:
# + Define a [`problem`](@ref problem) using your data.
# + Choose a [`basis`](@ref Basis).
#   + This is optional depending on which solver you choose.
# + [`Solve`](@ref solve) the problem.

using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra

# Generate a test problem 

f(u) = u .^ 2 .+ 2.0u .- 1.0
X = randn(1, 100);
Y = reduce(hcat, map(f, eachcol(X)));

# Create a problem from the data
problem = DirectDataDrivenProblem(X, Y, name = :Test)

# Choose a basis
@variables u
basis = Basis(monomial_basis([u], 2), [u])
println(basis)

# Solve the problem, using the solver of your choosing

res = solve(problem, basis, STLSQ())
println(res)
println(result(res))

#md # ## [Copy-Pasteable Code](@id getting_started_code)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
