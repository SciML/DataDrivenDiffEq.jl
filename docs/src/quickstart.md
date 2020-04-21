## Getting Started

Okay, lets start! In the following, we will use some of the techniques provide by `DataDrivenDiffEq` to
infer some models. To begin, lets create our own data for the linear pendulum with damping.

```@example
using OrdinaryDiffEq
using Plots

function pendulum!(du, u, p, t)
  du[1] = u[2]
  du[2] = -u[1] - 0.1*u[2]
end

u0 = Float64[0.99π; -0.3]
tspan = (0.0, 40.0)

problem = ODEProblem(pendulum!, u0, tspan)
solution = solve(problem, Tsit5(), saveat = 1.0)

plot(solution)
savefig("pendulum_solution.png") #hide
```
![](pendulum_solution.png)

Lets assume we have just the trajectory data and lets call it `X`.
Since we gathered the data at at fixed interval of one time unit, we will try to fit
a linear model. And of course, we use a subset of the data for training and the rest for
testing.

```@example
X = Array(sol)

approximation = DMD(X[:, 1:20])

approx_prob = DiscreteProblem(koopman, u0, tspan)
approx_sol = solve(approx_prob, FunctionMap())

plot(approx_sol, label = ["u[1]" "u[2]"]) #hide
scatter!(solution, label = ["True u[1]" "True u[2]"]) #hide
savefig("pendulum_approximation.png") #hide
```
![Approximation](pendulum_approximation.png)

Yeah! The model fits! But what exactly did we do?

Given the data `X` a [Dynamic Mode Decomposition]()
