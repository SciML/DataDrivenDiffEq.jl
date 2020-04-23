# Quickstart

In the following, we will use some of the techniques provide by `DataDrivenDiffEq` to infer some models.

## Linear Damped Oscillator

To begin, lets create our own data for the linear oscillator with damping.


```@example 1
using OrdinaryDiffEq
using Plots
gr()

using DataDrivenDiffEq
using LinearAlgebra

function linear!(du, u, p, t)
  du[1] = u[2]
  du[2] = -u[1] - 0.1*u[2]
end

u0 = Float64[0.99π; -0.3]
tspan = (0.0, 40.0)

problem = ODEProblem(linear!, u0, tspan)
solution = solve(problem, Tsit5(), saveat = 1.0)

plot(solution)
savefig("linear_solution.png") #hide
```
![](linear_solution.png)

Lets assume we have just the trajectory data and lets call it `X`.
Since we gathered the data at at fixed interval of one time unit, we will try to fit
a linear model. And of course, we use a subset of the data for training and the rest for
testing.

```@example 1

X = Array(solution)

approximation = DMD(X[:, 1:20])

approx_prob = DiscreteProblem(approximation, u0, tspan)
approx_sol = solve(approx_prob, FunctionMap())

plot(approx_sol, label = ["u[1]" "u[2]"]) #hide
scatter!(solution, label = ["True u[1]" "True u[2]"]) #hide
savefig("pendulum_approximation.png") #hide
```
![](pendulum_approximation.png)

Yeah! The model fits! But what exactly did we do?

`DMD` is short for [Dynamic Mode Decomposition](), a technique which generates a linear model from data. So given the data matrix `X` we simply divided it up into two data sets and performed a linear fitting between those.

Note that we fitted a **discrete** model which fits our **continuous** data. This is possible because

+ The measurements were taken at an interval of `1.0`
+ The original, unknown model has a discrete, linear solution

To check this, we can compare the `operator` of our linear fit with the matrix exponential of the original model.

```@example 1
dt = 1.0
K = operator(approximation)
norm(K - exp(dt*[0.0 1.0; -1.0 -0.1]), 2)
```

The reason for using `operator` as a function to get the corresponding matrix of the approximation is the connection of Dynamic Mode Decomposition to the [Koopman Operator](). You might have noticed that the return value of `DMD` is a `LinearKoopman`.

The `LinearKoopman` overloads some useful functions from `LinearAlgebra` to perform analysis. Lets have a look at the eigenvalues of the operator

```@example 1
scatter(eigvals(approximation))

# Add the stability margin
ϕ = 0:0.01π:2π
plot!(cos.(ϕ), sin.(ϕ),
  color = :red, linestyle = :dot,
  label = "Stability Margin",
  xlim = (-1,1), ylim = (-1,1), legend = :bottomleft)

savefig("eigenvalue_lineardamped.png") #hide
```
![](eigenvalue_lineardamped.png)

For more information on the `LinearKoopman` have a look at the corresponding documentation.

But wait! We want a continuous model. There is also a corresponding algorithm for this : `gDMD` !
Opposed to `DMD` which provides a discrete model based on the direct measurements `X`, `gDMD` estimates the generator of the dynamical system given `X` and the differential states `DX`. Since we did not measure any differential states, we can just provide a vector of time measurements. `gDMD` will automatically interpolate using [DataInterpolations.jl](https://github.com/PumasAI/DataInterpolations.jl) and perform numerical differentiation using [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl).

Here, we will provide `gDMD` with the measurement data and use a new sample time of `0.1`

```@example 1
t = solution.t
X = Array(solution)

generator_approximation = gDMD(t[1:20], X[:, 1:20], dt = 0.1)

generator_prob = ODEProblem(generator_approximation, u0 , tspan)
generator_sol = solve(generator_prob, Tsit5())

plot(generator_sol, label = ["u[1]" "u[2]"]) #hide
scatter!(solution, label = ["True u[1]" "True u[2]"]) #hide
savefig("linear_approximation_cont.png") #hide
```
![](linear_approximation_cont.png)

Since we have a continuous estimation, lets look at the `generator` of the estimation

```@example 1
G = generator(generator_approximation)
norm(G-[0.0 1.0; -1.0 -0.1], 2)
```
