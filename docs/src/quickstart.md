# Quickstart

In the following, we will use some of the techniques provided by `DataDrivenDiffEq` to infer some models.

## Linear Damped Oscillator - Dynamic Mode Decomposition

To begin, let's create our own data for the linear oscillator with damping.


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

Let's assume we have just the trajectory data and let's call it `X`.
Since we gathered the data at a fixed interval of one time unit, we will try to fit
a linear model. And, of course, we use a subset of the data for training and the rest for
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

`DMD` is short for [Dynamic Mode Decomposition](@ref), a technique which generates a linear model from data. So, given the data matrix `X`, we simply divided it up into two data sets and performed a linear fitting between those.

Note that we fitted a **discrete** model, which fits our **continuous** data. This is possible because:

+ The measurements were taken at an interval of `1.0`
+ The original, unknown model has a discrete, linear solution

To check this, we can compare the `operator` of our linear fit with the matrix exponential of the original model.

```@example 1
dt = 1.0
K = operator(approximation)
norm(K - exp(dt*[0.0 1.0; -1.0 -0.1]), 2)
```

The reason for using `operator` as a function to get the corresponding matrix of the approximation is the connection of Dynamic Mode Decomposition to the [Koopman Operator](@ref koopman_operator). You might have noticed that the return value of `DMD` is a `LinearKoopman`.

The `LinearKoopman` overloads some useful functions from `LinearAlgebra` to perform analysis. Let's have a look at the eigenvalues of the operator:

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

For more information on the `LinearKoopman`, have a look at the corresponding documentation.

But wait! We want a continuous model. There is also a corresponding algorithm for this : `gDMD` !
As opposed to `DMD`, which provides a discrete model based on the direct measurements `X`, `gDMD` estimates the generator of the dynamical system given `X` and the differential states `DX`. Since we did not measure any differential states, we can just provide a vector of time measurements. `gDMD` will automatically interpolate using [DataInterpolations.jl](https://github.com/PumasAI/DataInterpolations.jl) and perform numerical differentiation using [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl).

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

Since we have a continuous estimation, let's look at the `generator` of the estimation

```@example 1
G = generator(generator_approximation)
norm(G-[0.0 1.0; -1.0 -0.1], 2)
```

## Nonlinear Systems - Extended Dynamic Mode Decomposition

But what about nonlinear systems? Even though Dynamic Mode Decomposition will help us
to figure out the *best linear fit*, we are interested in figuring out all the nonlinear parts of the equations.
Luckily, Koopman theory covers this! To put it very (very very) simply : If you spread out your information in many **observable functions**, you will end up with a linear system in those observables. So you might end up with a trade-off between a huge system which is linear in the observables vs a small, nonlinear system.

But how can we leverage this? We use the [Extended Dynamic Mode Decomposition](https://arxiv.org/abs/1408.4408), or `EDMD` for short.
`EDMD` does more or less the exact same thing like `DMD`, but in the new `Basis` of nonlinear observables.
We will investigate now a fairly standard system, with a slow and fast manifold, for which there exists an [analytical solution of this problem](https://arxiv.org/abs/1510.03007).

```@example 2
using OrdinaryDiffEq
using Plots
gr()

using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit

function slow_manifold(du, u, p, t)
  du[1] = p[1]*u[1]
  du[2] = p[2]*(u[2]-u[1]^2)
end

u0 = [3.0; -2.0]
tspan = (0.0, 10.0)
p = [-0.05, -1.0]

problem = ODEProblem(slow_manifold, u0, tspan, p)
solution = solve(problem, Tsit5(), saveat = 0.2)

X = Array(solution)
DX = solution(solution.t, Val{1})

plot(solution) # hide
savefig("slow_manifold.png") # hide
```
![](slow_manifold.png)

Since we want to estimate the continuous system, we also capture the trajectory of the differential states.
Now, we will create our nonlinear observables, which is represented as a `Basis` in `DataDrivenDiffEq.jl`.

```@example 2
@variables u[1:2]

observables = [u; u[1]^2]

basis = Basis(observables, u)
```

A `Basis` captures a bunch of functions defined over some variables provided via [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl).
Here, we included the state and `u[1]^2`. Now, we simply call `gEDMD`, which
will compute the generator of the [Koopman Operator](@ref koopman_operator) associated with the model.

```@example 2
approximation = gEDMD(X, DX, basis)

approximation_problem = ODEProblem(approximation, u0, tspan)
generator_sol = solve(approximation_problem, Tsit5(), saveat = solution.t)

plot(generator_sol, label = ["u[1]" "u[2]"]) #hide
scatter!(solution, label = ["True u[1]" "True u[2]"]) #hide
savefig("slow_approximation_cont.png") #hide
scatter(eigvals(approximation), label = "Estimate") # hide
scatter!(eigvals([p[1] 0 0; 0 p[2] -p[2]; 0 0 2*p[1]]), label = "True", legend = :bottomright) #hide
savefig("eigenvalue_slowmanifold.png") #hide

```
![](slow_approximation_cont.png)

Looking at the eigenvalues of the system, we see that the estimated eigenvalues of the linear system are close to the true values.

![](eigenvalue_slowmanifold.png)

## Nonlinear Systems - Sparse Identification of Nonlinear Dynamics

Okay, so far we can fit linear models via DMD and nonlinear models via EDMD. But what if we want to find a model of a nonlinear system *without moving to Koopman space*? Simple, we use [Sparse Identification of Nonlinear Dynamics](https://www.pnas.org/content/113/15/3932) or `SINDy`.

As the name suggests, `SINDy` finds the sparsest basis of functions which build the observed trajectory. Again, we will start with a nonlinear system

```@example 3
using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra
using Plots
gr()

function pendulum(u, p, t)
    x = u[2]
    y = -9.81sin(u[1]) - 0.1u[2]
    return [x;y]
end

u0 = [0.4π; 1.0]
tspan = (0.0, 20.0)
problem = ODEProblem(pendulum, u0, tspan)
solution = solve(problem, Tsit5(), atol = 1e-8, rtol = 1e-8, saveat = 0.001)

X = Array(solution)
DX = solution(solution.t, Val{1})

plot(solution) # hide
savefig("nonlinear_pendulum.png") # hide
```
![](nonlinear_pendulum.png)

which is the simple nonlinear pendulum with damping.

Suppose we are like John and know nothing about the system, we have just the data in front of us. To apply `SINDy`, we need three ingredients:

+ A `Basis` containing all possible candidate functions which might be in the model
+ An optimizer which is able to produce a sparse output
+ A threshold for the optimizer

**It might seem to you that the third point is more a parameter of the optimizer (which it is), but, nevertheless, it is a crucial decision where to cut off parameters.**

So, let's create a bunch of basis functions for our problem first

```@example 3

@variables u[1:2]

h = Operation[u; u.^2; u.^3; sin.(u); cos.(u); 1]

basis = Basis(h, u)
nothing # hide
```

`DataDrivenDiffEq` comes with some optimizers to tackle sparse regression problems. Here, we will use `SR3`, used [here](https://arxiv.org/abs/1906.10612) and introduced [here](https://ieeexplore.ieee.org/document/8573778). We choose a threshold of `3.5e-1` and start the optimizer.

```@example 3
opt = SR3(3e-1, 1.0)
Ψ = SINDy(X[:, 1:1000], DX[:, 1:1000], basis, opt, maxiter = 10000, normalize = true)
print_equations(Ψ) # hide
```

We recovered the equations! Let's transform the `SINDyResult` into a performant piece of
Julia Code using `ODESystem`

```@example 3
sys = ODESystem(Ψ)
p = parameters(Ψ)

dudt = ODEFunction(sys)

estimator = ODEProblem(dudt, u0, tspan, p)
estimation = solve(estimator, Tsit5(), saveat = solution.t)

plot(solution.t[1:1000], solution[:,1:1000]', color = :red, line = :dot, label = nothing) # hide
plot!(solution.t[1000:end], solution[:,1000:end]', color = :blue, line = :dot,label = nothing) # hide
plot!(estimation, color = :green, label = "Estimation") # hide
savefig("SINDy_estimation.png") # hide
```
![](SINDy_estimation.png)
