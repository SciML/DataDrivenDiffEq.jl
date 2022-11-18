```@meta
EditURL = "<unknown>/docs/src/libs/datadrivendmd/example_01.jl"
```

# [Linear Time Discrete System](@id linear_discrete)

We will start by estimating the underlying dynamical system of a time discrete process based on some measurements via [Dynamic Mode Decomposition](https://arxiv.org/abs/1312.0041) on a simple linear system of the form ``u(k+1) = A u(k)``.

At first, we simulate the correspoding system using `OrdinaryDiffEq.jl` and generate a [`DiscreteDataDrivenProblem`](@ref DataDrivenProblem) from the simulated data.

````@example example_01
using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenDMD
using Plots

A = [0.9 -0.2; 0.0 0.2]
u0 = [10.0; -10.0]
tspan = (0.0, 11.0)

f(u,p,t) = A*u

sys = DiscreteProblem(f, u0, tspan)
sol = solve(sys, FunctionMap());
nothing #hide
````

Next we transform our simulated solution into a [`DataDrivenProblem`](@ref). Given that the solution knows its a discrete solution, we can simply write

````@example example_01
prob = DataDrivenProblem(sol)
````

And plot the solution and the problem

````@example example_01
plot(sol, label = string.([:x₁ :x₂]))
scatter!(prob)
````

To estimate the underlying operator in the states ``x_1, x_2``, we `solve` the estimation problem using the [`DMDSVD`](@ref) algorithm for approximating the operator. First, we will have a look at the [`DataDrivenSolution`](@ref)

````@example example_01
res = solve(prob, DMDSVD(), digits = 1)
````

We see that the system has been recovered correctly, indicated by the small error and high AIC score of the result. We can confirm this by looking at the resulting [`Basis`](@ref)

````@example example_01
get_basis(res)
````

And also plot the prediction of the recovered dynamics

````@example example_01
plot(res)
````

## [Copy-Pasteable Code](@id linear_discrete_copy_paste)

```julia
using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenDMD

A = [0.9 -0.2; 0.0 0.2]
u0 = [10.0; -10.0]
tspan = (0.0, 11.0)

f(u,p,t) = A*u

sys = DiscreteProblem(f, u0, tspan)
sol = solve(sys, FunctionMap());

prob = DataDrivenProblem(sol)

res = solve(prob, DMDSVD(), digits = 1)

get_basis(res)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

