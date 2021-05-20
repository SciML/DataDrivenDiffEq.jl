# Quickstart

In the following, we will use some of the techniques provided by `DataDrivenDiffEq` to infer some models.

## Nonlinear Systems - Sparse Identification of Nonlinear Dynamics

Assume you have a set of measurements and want to find the underlying continuous, nonlinear dynamical system. The answer is : [Sparse Identification of Nonlinear Dynamics](https://www.pnas.org/content/113/15/3932) or `SINDy`.

As the name suggests, `SINDy` finds the sparsest basis of functions which build the observed trajectory. Again, we will start with a nonlinear system

```@example 1
using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using Plots
using OrdinaryDiffEq
using Random

Random.seed!(1111) # Due to the noise

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
To estimate the system, we first create a `DataDrivenProblem` via feeding in the measurement data.
Using different collocation techniques, it automatically provides the derivative. Additional control signals can be passed
in as a function `(u,p,t)->control` or an array of measurements.

```@example 1
prob = ContinuousDataDrivenProblem(X, ts, GaussianKernel() ,
    U = (u,p,t)->[exp(-((t-5.0)/5.0)^2)], p = ones(2))

p1 = plot(ts, X', label = ["Measurement" nothing], color = :black, style = :dash, legend = :bottomleft, ylabel ="Measurement") # hide
plot!(ts, prob.X', label = ["Smoothed" nothing], color = :red) # hide
p2 = plot(ts, prob.DX', label = nothing, color = :red, ylabel = "Derivative") # hide
plot!(ts, DX', label = nothing, color = :black, style = :dash) # hide
p3 = plot(ts, prob.U', label = nothing, color = :red, xlabel = "Time [s]", ylabel = "Control") # hide
plot(p1,p2,p3, layout = (3,1), size = (600,600)) # hide
savefig("SINDy_Example_Data.png") # hide
```
![](SINDy_Example_Data.png)

Now we infer the systems structure. First we define a [`Basis`](@ref) which collects all possible candidate terms.
Since we want to use `SINDy`, we call `solve` with an `Optimizer`, in this case [`STLSQ`](@ref) which iterates different sparsity thresholds
and returns a pareto optimal solution. Note that we include the control signal in the basis as an additional variable `c`.
```@example 1
@variables u[1:2] c[1:1]
@parameters w[1:2]
h = Num[sin(w[1]*u[1]);cos(w[2]*u[1]); polynomial_basis(u, 5); c]
basis = Basis(h, u, parameters = w, controls = c)
λs = exp10.(-10:0.1:-1)
opt = STLSQ(λs)
res = solve(prob, basis, opt, progress = false, denoise = false, normalize = false, maxiter = 5000)
println(res) # hide
```
Where the resulting `SparseIdentificationResult` stores information about the infered model and the parameters:

```@example 1
system = result(res);
params = parameters(res);
println(system) #hide
println(params) #hide
```

which is indeed our pendulum model with a slight offset due to the noisy measurements and the estimation of the time derivates.

Since any system obtained via a `solve` command is a [`Basis`](@ref) and hence a subtype of an `AbstractSystem` defined in `ModelingToolkit`, we can simply simulate the result via:

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


## Implicit Nonlinear Dynamics

```@example 2
using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using Plots
using OrdinaryDiffEq

function michaelis_menten(u, p, t)
    [0.6 - 1.5u[1]/(0.3+u[1])]
end

u0 = [0.5]

problem_1 = ODEProblem(michaelis_menten, u0, (0.0, 4.0))
solution_1 = solve(problem_1, Tsit5(), saveat = 0.1)
problem_2 = ODEProblem(michaelis_menten, 2*u0, (4.0, 8.0))
solution_2 = solve(problem_2, Tsit5(), saveat = 0.1)
X = [solution_1[:,:] solution_2[:,:]]
ts = [solution_1.t; solution_2.t]

DX = similar(X)
for (i, xi) in enumerate(eachcol(X))
    DX[:, i] = michaelis_menten(xi, [], ts[i])
end

p1 = plot(ts, X', label = ["Measurement" nothing], color = :black, style = :dash, legend = :bottomleft, ylabel ="Measurement") # hide
p2 = plot(ts, DX', label = nothing, color = :black, style = :dash, ylabel = "Derivative", xlabel = "Time [s]") # hide
plot(p1,p2, layout = (2,1), size = (600,400)) # hide
savefig("SINDy_Example_Data_2.png") # hide

@parameters t
@variables u[1:2]
h = [monomial_basis(u[1:1], 4)...]
basis = Basis([h; h .* u[2]], u);
println(basis) # hide
```

![](SINDy_Example_Data_2.png)

```@example 2
prob = ContinuousDataDrivenProblem(X, ts, DX = DX)

opt = ImplicitOptimizer(2e-1)

res = solve(prob, basis, opt, normalize = false, denoise = false, maxiter = 1000);
println(res) # hide
```

```@example 2
system = result(res); # hide
println(system)
```

```@example 2
infered_prob = ODEAProblem(system, u0, tspan, parameters(res))
infered_solution = solve(infered_prob, Tsit5(), saveat = ts)
plot(infered_solution, label = ["Infered" nothing], color = :red) # hide
savefig("SINDy_Example_Data_Infered_2.png") # hide
```

![](SINDy_Example_Data_Infered_2.png)
