```@meta
EditURL = "<unknown>/docs/src/libs/datadrivendmd/example_02.jl"
```

# [Linear Time Continuous System](@id linear_continuous)

Similar to the [`linear time discrete example`](@ref linear_discrete), we will now estimate a linear time continuous system ``\partial_t u = A u``.
We simulate the correspoding system using `OrdinaryDiffEq.jl` and generate a [`ContinuousDataDrivenProblem`](@ref DataDrivenProblem) from the simulated data.

````@example example_02
using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenDMD
using Plots

A = [-0.9 0.2; 0.0 -0.2]
u0 = [10.0; -10.0]
tspan = (0.0, 10.0)

f(u,p,t) = A*u

sys = ODEProblem(f, u0, tspan)
sol = solve(sys, Tsit5(), saveat = 0.05);
nothing #hide
````

We could use the `DESolution` to define our problem, but here we want to use the data for didactic purposes.
For a [`ContinuousDataDrivenProblem`](@ref DataDrivenProblem), we need either the state trajectory and the timepoints or the state trajectory and its derivate.

````@example example_02
X = Array(sol)
t = sol.t
prob = ContinuousDataDrivenProblem(X, t)
````

And plot the problems data.

````@example example_02
plot(prob)
````

We can see that the derivative has been automatically added via a [`collocation`](@ref collocation) method, which defaults to a `LinearInterpolation`.
We can do a visual check and compare our derivatives with the interpolation of the `ODESolution`.

````@example example_02
DX = Array(sol(t, Val{1}))
scatter(t, DX', label = ["Solution" nothing], color = :red, legend = :bottomright)
plot!(t, prob.DX', label = ["Linear Interpolation" nothing], color = :black)
````

Since we have a linear system, we can use `gDMD`, which approximates the generator of the dynamics

````@example example_02
res = solve(prob, DMDSVD())
println(res)
````

And also plot the prediction of the recovered dynamics

````@example example_02
plot(res)
````

## [Copy-Pasteable Code](@id linear_continuous_copy_paste)

```julia
using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenDMD

A = [-0.9 0.2; 0.0 -0.2]
u0 = [10.0; -10.0]
tspan = (0.0, 10.0)

f(u,p,t) = A*u

sys = ODEProblem(f, u0, tspan)
sol = solve(sys, Tsit5(), saveat = 0.05);

X = Array(sol)
t = sol.t
prob = ContinuousDataDrivenProblem(X, t)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

