```@meta
EditURL = "<unknown>/docs/src/libs/datadrivendmd/example_04.jl"
```

# [Nonlinear Time Continuous System](@id nonlinear_continuos)

Similarly, we can use the [Extended Dynamic Mode Decomposition](https://link.springer.com/article/10.1007/s00332-015-9258-5) via a nonlinear [`Basis`](@ref) of observables. Here, we will look at a rather [famous example](https://arxiv.org/pdf/1510.03007.pdf) with a finite dimensional solution.

````@example example_04
using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenDMD
using Plots

function slow_manifold(du, u, p, t)
    du[1] = p[1] * u[1]
    du[2] = p[2] * (u[2] - u[1]^2)
end

u0 = [3.0; -2.0]
tspan = (0.0, 5.0)
p = [-0.8; -0.7]

problem = ODEProblem{true, SciMLBase.NoSpecialize}(slow_manifold, u0, tspan, p)
solution = solve(problem, Tsit5(), saveat = 0.1);
plot(solution)
````

Since we are dealing with a continuous system in time, we define the associated [`DataDrivenProblem`](@ref) accordingly using the measured states `X`, their derivatives `DX` and the time `t`.

````@example example_04
prob = DataDrivenProblem(solution)
plot(prob)
````

Additionally, we need to define the [`Basis`](@ref) for our lifting, before we `solve` the problem in the lifted space.

````@example example_04
@parameters t
@variables u(t)[1:2]
Ψ = Basis([u; u[1]^2], u, independent_variable = t)
res = solve(prob, Ψ, DMDPINV(), digits = 2)
println(res) #hide
````

We can also use different metrics on the `DataDrivenSolution` like the `aic`

````@example example_04
aic(res)
````

The `aicc`

````@example example_04
aicc(res)
````

The `bic`

````@example example_04
bic(res)
````

The `loglikelihood`

````@example example_04
loglikelihood(res)
````

And the number of parameters

````@example example_04
dof(res)
````

Lets have a closer look at the `Basis`

````@example example_04
basis = get_basis(res)
println(basis) #hide
````

And the connected parameters

````@example example_04
get_parameter_map(basis)
````

And plot the results

````@example example_04
plot(res)
````

## [Copy-Pasteable Code](@id linear_discrete_copy_paste)

```julia
using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenDMD

function slow_manifold(du, u, p, t)
    du[1] = p[1] * u[1]
    du[2] = p[2] * (u[2] - u[1]^2)
end

u0 = [3.0; -2.0]
tspan = (0.0, 5.0)
p = [-0.8; -0.7]

problem = ODEProblem{true, SciMLBase.NoSpecialize}(slow_manifold, u0, tspan, p)
solution = solve(problem, Tsit5(), saveat = 0.1);

prob = DataDrivenProblem(solution)

@parameters t
@variables u(t)[1:2]
Ψ = Basis([u; u[1]^2], u, independent_variable = t)
res = solve(prob, Ψ, DMDPINV(), digits = 2)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

