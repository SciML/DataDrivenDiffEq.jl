# Nonlinear Systems

Estimation examples for nonlinear systems.

## Nonlinear System with Extended Dynamic Mode Decomposition

Similarly, we can use the [Extended Dynamic Mode Decomposition](https://link.springer.com/article/10.1007/s00332-015-9258-5) via a nonlinear [`Basis`](@ref) of observables. Here, we will look at a rather [famous example](https://arxiv.org/pdf/1510.03007.pdf) with a finite dimensional solution.

```@example 3
using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using OrdinaryDiffEq
using Plots

function slow_manifold(du, u, p, t)
    du[1] = p[1] * u[1]
    du[2] = p[2] * (u[2] - u[1]^2)
end

u0 = [3.0; -2.0]
tspan = (0.0, 5.0)
p = [-0.8; -0.7]

problem = ODEProblem(slow_manifold, u0, tspan, p)
solution = solve(problem, Tsit5(), saveat = 0.01)
plot(solution) # hide
savefig("EDMD_Example_1.png") # hide
```
![](EDMD_Example_1.png)

Since we are dealing with a continuous system in time, we define the associated [`DataDrivenProblem`](@ref) accordingly using the measured states `X`, their derivatives `DX` and the time `t`.

```@example 3
prob = ContinuousDataDrivenProblem(solution)
```
Additionally, we need to define the [`Basis`](@ref) for our lifting, before we `solve` the problem in the lifted space.

```@example 3
@variables u[1:2]
Ψ = Basis([u; u[1]^2], u)
res = solve(prob, Ψ, DMDPINV(), digits = 1)
system = result(res)
println(res) # hide
println(system) # hide
println(parameters(res)) # hide
```

The underlying dynamics have been recovered correctly by the algorithm!

The eigendecomposition of the (generator of the) Koopman operator can be accessed via [`generator`](@ref).

```@example 3
generator(system)
```

## Nonlinear Systems - Sparse Identification of Nonlinear Dynamics

To find the underlying system without any [`Algorithms`](@ref koopman_algorithms) related to Koopman operator theory, we can use  [Sparse Identification of Nonlinear Dynamics](https://www.pnas.org/content/113/15/3932) - SINDy for short. As the name suggests, it finds the sparsest basis of functions which build the observed trajectory. Again, we will start with a nonlinear system:

```@example 1
using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using OrdinaryDiffEq
using Plots
using Random
using Symbolics: scalarize

Random.seed!(1111) # For the noise

# Create a nonlinear pendulum
function pendulum(u, p, t)
    x = u[2]
    y = -9.81sin(u[1]) - 0.3u[2]^3 -3.0*cos(u[1]) - 10.0*exp(-((t-5.0)/5.0)^2)
    return [x;y]
end

u0 = [0.99π; -1.0]
tspan = (0.0, 15.0)
prob = ODEProblem(pendulum, u0, tspan)
sol = solve(prob, Tsit5(), saveat = 0.01)

# Create the data with additional noise
X = sol[:,:] + 0.1 .* randn(size(sol))
DX = similar(sol[:,:])

for (i, xi) in enumerate(eachcol(sol[:,:]))
    DX[:,i] = pendulum(xi, [], sol.t[i])
end

ts = sol.t
nothing #hide
```

To estimate the system, we first create a [`DataDrivenProblem`](@ref) via feeding in the measurement data.
Using a [Collocation](@ref) method, it automatically provides the derivative. Control signals can be passed
in as a function `(u,p,t)->control` or an array of measurements.

```@example 1
prob = ContinuousDataDrivenProblem(X, ts, GaussianKernel() ,
    U = (u,p,t)->[exp(-((t-5.0)/5.0)^2)], p = ones(2))

# Lets have a look at the data defining the problem
p_prob = plot(prob, size = (600,600))
savefig("SINDY_Example_Data.png") # hide
```
![](SINDy_Example_Data.png)

Now we infer the system structure. First we define a [`Basis`](@ref) which collects all possible candidate terms.
Since we want to use SINDy, we call `solve` with an [`Optimizer`](@ref sparse_optimization), in this case [`STLSQ`](@ref) which iterates different sparsity thresholds
and returns a pareto optimal solution of the underlying [`sparse_regression!`](@ref). Note that we include the control signal in the basis as an additional variable `c`.

```@example 1
@variables u[1:2] c[1:1]
@parameters w[1:2]
u = scalarize(u)
c = scalarize(c)
w = scalarize(w)

h = Num[sin.(w[1].*u[1]);cos.(w[2].*u[1]); polynomial_basis(u, 5); c]

basis = Basis(h, u, parameters = w, controls = c)

λs = exp10.(-10:0.1:-1)
opt = STLSQ(λs)
res = solve(prob, basis, opt, progress = false, denoise = false, normalize = false, maxiter = 5000)
println(res) # hide
```

!!! info
    A more detailed description of the result can be printed via `print(res, Val{true})`, which also includes the discovered equations and parameter values.

Where the resulting [`DataDrivenSolution`](@ref) stores information about the inferred model and the parameters:

```@example 1
system = result(res);
params = parameters(res);
println(system) #hide
println(params) #hide
```

And a visual check of the result can be perfomed via plotting the result

```@example 1
plot(res)
savefig("SINDy_Result_Example1.png") # hide
```
![](SINDy_Result_Example1.png)

Since any system obtained via a `solve` command is a [`Basis`](@ref) and hence a subtype of an `AbstractSystem` defined in [`ModelingToolkit`](https://github.com/SciML/ModelingToolkit.jl), we can simply simulate the result via:

```@example 1
infered_prob = ODEProblem(system, u0, tspan, parameters(res))
infered_solution = solve(infered_prob, Tsit5(), saveat = ts)
plot(infered_solution, label = ["Infered" nothing], color = :red) # hide

function pendulum(u, p, t) # hide
    x = u[2] # hide
    y = -9.81sin(u[1]) - 0.3u[2]^3 -3.0*cos(u[1]) # hide
    return [x;y] # hide
end # hide

prob = ODEProblem(pendulum, u0, tspan) # hide
sol = solve(prob, Tsit5(), saveat = 0.01) # hide

plot!(sol, label = ["Ground Truth" nothing], color = :black, style = :dash) # hide
savefig("SINDy_Example_Data_Infered.png") #hide
```

!!! warning
    As of now, the control input is dropped in the simulation of a system. We are working on this and pull requests are welcome!

![](SINDy_Example_Data_Infered.png)

As we can see above, the estimated system matches the ground truth reasonably well.
