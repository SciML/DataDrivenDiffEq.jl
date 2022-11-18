```@meta
EditURL = "<unknown>/docs/src/libs/datadrivensparse/example_01.jl"
```

# [Getting Started](@id getting_started)

The workflow for [DataDrivenDiffEq.jl](https://github.com/SciML/DataDrivenDiffEq.jl) is similar to other [SciML](https://sciml.ai/) packages.
You start by defining a [`DataDrivenProblem`](@ref) and then dispatch on the [`solve`](@ref solve) command to return a [`DataDrivenSolution`](@ref).

Here is an outline of the required elements and choices:
+ Define a [`DataDrivenProblem`](@ref problem) using your data.
+ Optional: Choose a [`Basis`](@ref).
+ [`solve`](@ref solve) the problem.

````@example example_01
using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra
using DataDrivenSparse
````

Generate a test problem

````@example example_01
f(u) = u.^2 .+ 2.0u .- 1.0
X = randn(1, 100);
Y = reduce(hcat, map(f, eachcol(X)));
nothing #hide
````

Create a problem from the data

````@example example_01
problem = DirectDataDrivenProblem(X, Y, name = :Test)
````

Choose a basis

````@example example_01
@variables u
basis = Basis(monomial_basis([u], 2), [u])
println(basis) # hide
````

Solve the problem, using the solver of your choosing

````@example example_01
res = solve(problem, basis, STLSQ())
println(res) # hide
````

## [Copy-Pasteable Code](@id getting_started_code)

```julia
using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra
using DataDrivenSparse

f(u) = u.^2 .+ 2.0u .- 1.0
X = randn(1, 100);
Y = reduce(hcat, map(f, eachcol(X)));

problem = DirectDataDrivenProblem(X, Y, name = :Test)

@variables u
basis = Basis(monomial_basis([u], 2), [u])
println(basis) # hide

res = solve(problem, basis, STLSQ())
println(res) # hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

