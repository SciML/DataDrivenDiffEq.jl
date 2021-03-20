# Quickstart

In the following, we will use some of the techniques provided by `DataDrivenDiffEq` to infer some models.

## Nonlinear Systems - Sparse Identification of Nonlinear Dynamics

Okay, so far we can fit linear models via DMD and nonlinear models via EDMD. But what if we want to find a model of a nonlinear system *without moving to Koopman space*? Simple, we use [Sparse Identification of Nonlinear Dynamics](https://www.pnas.org/content/113/15/3932) or `SINDy`.

As the name suggests, `SINDy` finds the sparsest basis of functions which build the observed trajectory. Again, we will start with a nonlinear system

```@example 3
using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using Plots
using OrdinaryDiffEq
using DataDrivenDiffEq.Optimize

# Create a nonlinear pendulum
function pendulum(u, p, t)
    x = u[2]
    y = -9.81sin(u[1]) - 0.3u[2]^3 -3.0*cos(u[1]) - 10.0*exp(-((t-5.0)/5.0)^2)
    return [x;y]
end

u0 = [0.99π; -1.0]
tspan = (0.0, 15.0)
prob = ODEProblem(pendulum, u0, tspan)
sol = solve(prob, Tsit5(), saveat = 0.1)

# Create the data
X = sol[:,:] .+ 0.2*randn(size(sol)...)
DX = similar(sol[:,:])
for (i, xi) in enumerate(eachcol(sol[:,:]))
    DX[:,i] = pendulum(xi, [], sol.t[i])
end
ts = sol.t
```
To estimate the system, we first create a `DataDrivenProblem` via feeding in the measurement data.
Using different collocation techniques, it automatically provides the derivative. Additional control signals can be passed
in as a function `(u,p,t)->control` or an array of measurements.
```@example 3
prob = ContinuousDataDrivenProblem(X, ts, GaussianKernel(),U = (u,p,t)->[exp(-((t-5.0)/5.0)^2)],p = ones(2))
p1 = scatter(ts, X', label = ["True" nothing], color = :black, legend = :bottomleft, ylabel ="Measurement") # hide
plot!(ts, prob.X', label = ["Smoothed" nothing], color = :red) # hide
p2 = plot(ts, prob.DX', label = nothing, color = :red, ylabel = "Derivative") # hide
scatter!(ts, DX', label = nothing, color = :black) # hide
p3 = plot(ts, prob.U', label = nothing, color = :red, xlabel = "Time [s]", ylabel = "Control") # hide
plot(p1,p2,p3, layout = (3,1), size = (600,600)) # hide
savefig("SINDy_Example_Data.png") # hide
```
![](SINDy_Example_Data.png)

Now we infer the systems structure. First we define a `Basis` which collects all possible candidate terms.
Since we want to use `SINDy`, we call `solve` with an `Optimizer`, in this case `ADMM` which iterates different sparsity thresholds
and returns a pareto optimal solution. Note that we include the control signal in the basis as an additional variable `c`.
```@example 3
@variables u[1:2] c[1:1]
@parameters w[1:2]
h = Num[sin(w[1]*u[1]);cos(w[2]*u[1]); polynomial_basis(u, 5); c]
basis = Basis(h, u, parameters = w, controls = c)
λs = exp10.(-10:1:10)
opt = ADMM(λs)
res = solve(prob, basis, opt, progress = false, denoise = false, normalize = true, maxiter = 5000)
println(res.res) # hide
```

which is the simple nonlinear pendulum with damping.
