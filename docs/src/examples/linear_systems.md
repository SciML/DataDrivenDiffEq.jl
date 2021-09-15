# Linear Systems

Estimation examples for linear systems.

## Linear Systems via Dynamic Mode Decomposition

We will start by estimating the underlying dynamical system of a time discrete process based on some measurements via [Dynamic Mode Decomposition](https://arxiv.org/abs/1312.0041). We will model a simple linear system of the form ``u_{i+1} = A u_i``.

```@example 4
using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using OrdinaryDiffEq
using Plots # hide

A = [0.9 -0.2; 0.0 0.2]
u0 = [10.0; -10.0]
tspan = (0.0, 11.0)

f(u,p,t) = A*u

sys = DiscreteProblem(f, u0, tspan)
sol = solve(sys, FunctionMap())
plot(sol) # hide
savefig("DMD_Example_1.png") # hide
```
![](DMD_Example_1.png)

To estimate the underlying operator in the states ``u_1, u_2``, we simply define a discrete [`DataDrivenProblem`](@ref) using the measurements and time, and then `solve` the estimation problem using the [`DMDSVD`](@ref) algorithm for approximating the operator.

```@example 4

prob = DiscreteDataDrivenProblem(sol)

res = solve(prob, DMDSVD(), digits = 1)
system = result(res)
println(system) # hide
```

The [`DataDrivenSolution`](@ref) contains an explicit result which is a [`Koopman`](@ref), defining all necessary information, e.g. the associated operator (which corresponds to our matrix `A`).

```@example 4
Matrix(system)
```
In general, we can skip the expensive process of deriving a callable symbolic system and return just the basic definitions using the `operator_only` keyword.

```@example 4
res = solve(prob, DMDSVD(), digits = 1, operator_only = true)
```

Where `K` is the associated operator given as its eigendecomposition, `B` is a possible mapping of inputs onto the states, `C` is the linear mapping from the lifted observables back onto the original states and `Q` and `P` are used for updating the operator.
