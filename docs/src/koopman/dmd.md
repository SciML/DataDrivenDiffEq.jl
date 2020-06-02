# Dynamic Mode Decomposition

The (Exact) [Dynamic Mode Decomposition](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/dynamic-mode-decomposition-of-numerical-and-experimental-data/AA4C763B525515AD4521A6CC5E10DBD4) is a method for
generating an approximating linear differential equation directly from the observed data.
If `X` and `Y` are data matrices containing points of the same trajectory, than `DMD` approximates

```math
K = Y~X^{\dagger}
```

where ``\dagger`` denotes the Moore-Penrose pseudoinverse and `K` is the approximation of the [Koopman Operator](@ref koopman_operator).

`DMD` approximates *discrete time systems* of the form

```math
u_{i+1} = K ~ u_{i}
```

`gDMD` approximates *continuous time systems* of the form

```math
\frac{d}{dt}u =  K_{G} ~ u
```

where ``K_{G}`` is the generator of the [Koopman Operator](@ref koopman_operator).

## Functions

```@docs
DMD
gDMD
```

## Examples

```@example dmd_1
using DataDrivenDiffEq
using OrdinaryDiffEq
using LinearAlgebra
using Plots
gr()

function linear_discrete(du, u, p, t)
    du[1] = 0.9u[1]
    du[2] = 0.05u[2] + 0.1u[1]
end

u0 = [10.0; -2.0]
tspan = (0.0, 20.0)
problem = DiscreteProblem(linear_discrete, u0, tspan)
solution = solve(problem, FunctionMap())
```

```@example dmd_1
X = Array(solution)

approx = DMD(X[:,1:3])

prob_approx = DiscreteProblem(approx, u0, tspan)
approx_sol = solve(prob_approx, FunctionMap())

plot(approx_sol, label = ["u[1]" "u[2]"]) #hide
plot!(solution, label = ["True u[1]" "True u[2]"]) #hide
savefig("dmd_example_1.png") #hide
```

![](dmd_example_1.png)

```@example dmd_1

function linear_discrete_2(du, u, p, t)
    du[1] = 0.9u[1] + 0.05u[2]
    du[2] = 0.1u[1]
end

problem = DiscreteProblem(linear_discrete_2, u0, tspan)
solution = solve(problem, FunctionMap())

X = Array(solution)

update!(approx, X[:, 4:end-1], X[:, 5:end])

prob_approx = DiscreteProblem(approx, u0, tspan)
approx_sol = solve(prob_approx, FunctionMap())

# Show solutions
plot(approx_sol, label = ["u[1]" "u[2]"]) #hide
plot!(solution, label = ["True u[1]" "True u[2]"]) #hide
savefig("dmd_example_2.png") #hide
```
![](dmd_example_2.png)
