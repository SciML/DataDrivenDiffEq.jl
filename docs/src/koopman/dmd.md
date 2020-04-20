## Dynamic Mode Decomposition


#### (Exact) Dynamic Mode Decomposition (DMD)

Dynamic Mode Decomposition, or Exact Dynamic Mode Decomposition, is a method for
generating an approximating linear differential equation (the Koopman operator)
from the observed data.

To construct the approximation, use:

```julia
ExactDMD(X; dt = 0.0)
```

-  `X` is a data matrix of the observations over time, where each column is
  all observables at a given time point.
- `dt` is an optional value for the spacings
  between the observation timepoints which is used in the construction of the
  continuous dynamics.

#### Extended Dynamic Mode Decomposition (eDMD)

Extended Dynamic Mode Decomposition (eDMD), is a method for
generating an approximating linear differential equation (the Koopman operator)
for the observed data in a chosen basis of observables. Thus the signature is:

```julia
ExtendedDMD(X,basis; dt = 1.0)
```

-  `X` is a data matrix of the observations over time, where each column is
  all observables at a given time point.
- `dt` is an optional value for the spacings
  between the observation timepoints which is used in the construction of the
  continuous dynamics.

#### Shared DMD Features

To get the dynamics from the DMD object, use the `dynamics` function:

```julia
dynamics(dmd, discrete=true)
```

This will build the `f` function for DifferentialEquations.jl integrators, and
defaults to building approximations for `DiscreteProblem`, but this is changed
to approximating `ODEProblem`s by setting `discrete=false`.

The linear approximation can also be analyzed using the following functions:

```julia
eigen(dmd)
eigvals(dmd)
eigvecs(dmd)
```

Additionally, DMD objects can be updated to add new measurements via:

```julia
update!(dmd, x; Δt = 0.0, threshold = 1e-3)
```

## Examples

### Structural Identification with SInDy

In this example we will showcase how to automatically recover the differential
equations from data using the SInDy method. First, let's generate data
from the pendulum model. The pendulum model looks like:

```julia
using DataDrivenDiffEq, ModelingToolkit, DifferentialEquations, LinearAlgebra,
      Plots

function pendulum(u, p, t)
    x = u[2]
    y = -9.81sin(u[1]) - 0.1u[2]
    return [x;y]
end

u0 = [0.2π; -1.0]
tspan = (0.0, 40.0)
prob = ODEProblem(pendulum, u0, tspan)
sol_full = solve(prob,Tsit5())
sol = solve(prob,Tsit5(),saveat=0.75)
data = Array(sol)

plot(sol_full)
scatter!(sol.t,data')
```

In order to perform the SInDy method, we will need to get an approximate
derivative for each observable at each time point. We will do this with the
following helper function which fits 1-dimensional splines to each observable's
time series and uses the derivative of the splines:

```julia
using Dierckx
function colloc_grad(t::T, data::D) where {T, D}
  splines = [Dierckx.Spline1D(t, data[i,:]) for i = 1:size(data)[1]]
  grad = [Dierckx.derivative(spline, t[1:end]) for spline in splines]
  grad = [[grad[1][i],grad[2][i]] for i = 1:length(grad[1])]
  grad = convert(Array, VectorOfArray(grad))
  return grad
end
DX = colloc_grad(sol.t,data)
```

Now that we have the data, we need to choose a basis to fit it to. We know it's
a differential equation in two variables, so let's define our two symbolic
variables with ModelingToolkit:

```julia
@variables u[1:2]
```

Now let's choose a basis. We do this by building an array of ModelingToolkit
`Operation`s that represent the possible terms in our equation. Let's do this
with a bunch of polynomials, but also make sure to include some trigonometric
functions (since the true solution has trigonometric functions!):

```julia
# Lots of polynomials
polys = [u[1]^0]
for i ∈ 1:3
    for j ∈ 1:3
        push!(polys, u[1]^i*u[2]^j)
    end
end

# And some other stuff
h = [1u[1];1u[2]; cos(u[1]); sin(u[1]); u[1]*u[2]; u[1]*sin(u[2]); u[2]*cos(u[2]); polys...]
```

Now we build our basis:

```julia
basis = Basis(h, u)
```

From this we perform our SInDy to recover the differential equations in this basis:

```julia
opt = STRRidge(1e-10)
Ψ = SInDy(data, DX, basis, maxiter = 50, opt = opt)
```

From here we can use Latexify.jl to generate the LaTeX form of the outputted
equations via:

```julia

```

Wow, we recovered the equations! However, let's assume we didn't know the
analytical solution. What we would want to do is double check how good our
is. To do this, we can generate the dynamics and simulate to see how good
the regenerated dynamics fit the original data:

```julia
estimator = ODEProblem(dynamics(Ψ), u0, tspan)
sol_ = solve(estimator, saveat = sol.t)
plot(sol_)
scatter!(data)
```

### Linear Approximation of Dynamics with DMD

Lets start by creating some data from a given linear discrete system

```julia
function linear_discrete(du, u, p, t)
    du[1] = 0.9u[1]
    du[2] = 0.9u[2] + 0.1u[1]
end

u0 = [10.0; -2.0]
tspan = (0.0, 10.0)
prob = DiscreteProblem(linear_discrete, u0, tspan)
sol = solve(prob)
```

To approximate the system, we simply call

```julia
approx = ExactDMD(sol[:,:])
```

which returns us the approximation of the Koopman Operator.
As before, we can now get the dynamics and look at the approximation of our trajectory

```julia
approx_dudt = dynamics(approx)
prob_approx = DiscreteProblem(approx_dudt, u0, tspan)
approx_sol = solve(prob_approx)

plot(sol)
plot!(approx_sol)
```

But what about a differential equation? In contrast to `SInDy` `ExactDMD` does not require
differential data, but can estimate the dynamics from evenly sampled trajectories over time.
We pass that information via `dt` into the algorithm.

```julia
function linear(du, u, p, t)
    du[1] = -0.9*u[1] + 0.1*u[2]
    du[2] = -0.8*u[2]
end

prob_cont = ODEProblem(linear, u0, tspan)
sol_cont = solve(prob_cont, saveat = 0.1)

plot(sol_cont)

approx_cont = ExactDMD(sol_cont[:,:], dt = 0.1)
```
To get the continouos time dynamics, we simply use

```julia
test = dynamics(approx_cont, discrete = false)
```
and look at the results
```julia
approx_sys = ODEProblem(test, u0, tspan)
approx_sol = solve(approx_sys, saveat = 0.1)

plot(sol_cont)
plot!(approx_sol)
```
