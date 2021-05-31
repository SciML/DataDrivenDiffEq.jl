# Quickstart

In the following, we will use some of the techniques provided by `DataDrivenDiffEq` to infer some models.

## Linear Systems via Dynamic Mode Decomposition

We will start by estimating the underlying dynamical system of a time discrete process based on some measurements via [Dynamic Mode Decomposition](https://arxiv.org/abs/1312.0041). First, we model a simple linear system of the for ``u_{i+1} = A u_i``

```@example 4
using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using OrdinaryDiffEq
using Plots # hide

A = [0.9 -0.2; 0.0 0.2]
u0 = [10.0; -10.0]
tspan = (0.0, 11.0)

f(u,p,t) = A*u

sys = DiscreteProblem(f, u0, tspan)
sol = solve(sys, FunctionMap())
plot(sol) # hide
savefig("DMD_Example_1.png") # hide
```
![](DMD_Example_1.png)

To estimate the underlying operator in the states ``u_1, u_2``, we simply define a discrete [`DataDrivenProblem`](@ref) using the measurements and time and `solve` the estimation problem using the [`DMDSVD`](@ref) algorithm for approximating the operator.

```@example 4
X = Array(sol)

prob = DiscreteDataDrivenProblem(X, t = sol.t)

res = solve(prob, DMDSVD(), digits = 1)
system = result(res)
println(system) # hide
```

The [`DataDrivenSolution`](@ref) contains an explicit result which is a [`Koopman`](@ref), defining all necessary information, e.g. the associated operator (which corresponds to our abefore defined matrix ``A``).

```@example 4
Matrix(system)
```
In general, we can skip the expensive progress of deriving a callable symbolic system and return just the basic definitions using the `operator_only` keyword.

```@example 4
res = solve(prob, DMDSVD(), digits = 1, operator_only = true)
```

Where `K` is the associated operator given as its eigendecomposition, `B` is a possible mapping of inputs onto the states, `C` is the linear mapping from the lifted observeables back onto the original states and `Q` and `P` are used for updating the operator.

## Nonlinear System with Extended Dynamic Mode Decomposition

Similarly, we can use the [Extended Dynamic Mode Decomposition](https://link.springer.com/article/10.1007/s00332-015-9258-5) via a nonlinear [`Basis`](@ref) of observeables. Here, we look a rather [famous example](https://arxiv.org/pdf/1510.03007.pdf) with a finite dimensional solution.

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

Since we are dealing with an continuous system in time, we define the associated [`DataDrivenProblem`](@ref) accordingly using the measured states `X`, their derivates `DX` and the time `t`.

```@example 3
X = Array(solution)
t = solution.t
DX = solution(solution.t, Val{1})[:, :]

prob = ContinuousDataDrivenProblem(X, t, DX = DX)
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

The eigendecomposition of the Koopman operator can be accessed via [`operator`](@ref).

```@example 3
operator(system)
```

## Nonlinear Systems - Sparse Identification of Nonlinear Dynamics

To find the underlying system without any [`Algortihms`](@ref koopman_algorithms) related to Koopman operator theory, we can use  [Sparse Identification of Nonlinear Dynamics](https://www.pnas.org/content/113/15/3932) - SINDy for short. As the name suggests, it finds the sparsest basis of functions which build the observed trajectory. Again, we will start with a nonlinear system

```@example 1
using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using OrdinaryDiffEq
using Plots
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

To estimate the system, we first create a [`DataDrivenProblem`](@ref) via feeding in the measurement data.
Using a [Collocation](@ref) method, it automatically provides the derivative. Control signals can be passed
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

Now we infer the system structure. First we define a [`Basis`](@ref) which collects all possible candidate terms.
Since we want to use SINDy, we call `solve` with an [`Optimizer`](@id Sparse_Optimizers), in this case [`STLSQ`](@ref) which iterates different sparsity thresholds
and returns a pareto optimal solution of the underlying [`sparse_regression!`](@ref). Note that we include the control signal in the basis as an additional variable `c`.

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
Where the resulting [`DataDrivenSolution`](@ref) stores information about the infered model and the parameters:

```@example 1
system = result(res);
params = parameters(res);
println(system) #hide
println(params) #hide
```

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


## Implicit Nonlinear Dynamics : Michaelis Menten

But what if you want to estimate an implicitly defined system of the form ``f(u_t, u, p, t) = 0``?
Do not worry, since there exists a solution : Implicit Sparse Identification. It has been originally described in [this paper](http://ieeexplore.ieee.org/document/7809160/) and currently there exist [robust algorithms](https://royalsocietypublishing.org/doi/10.1098/rspa.2020.0279) to identify these systems.

We will focus on the [Michaelis Menten Kinetics](https://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics). As before, we will define the [`DataDrivenProblem`](@ref) and the [`Basis`](@ref) containing possible candidate functions for our [`sparse_regression!`](@ref).
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

prob = ContinuousDataDrivenProblem(X, ts, DX = DX)

p1 = plot(ts, X', label = ["Measurement" nothing], color = :black, style = :dash, legend = :bottomleft, ylabel ="Measurement") # hide
p2 = plot(ts, DX', label = nothing, color = :black, style = :dash, ylabel = "Derivative", xlabel = "Time [s]") # hide
plot(p1,p2, layout = (2,1), size = (600,400)) # hide
savefig("SINDy_Example_Data_2.png") # hide

@parameters t
D = Differential(t)
@variables u[1:1](t)
h = [monomial_basis(u[1:1], 4)...]
basis = Basis([h; h .* D(u[1])], [u; D(u[1])], iv = t)
println(basis) # hide
```

![](SINDy_Example_Data_2.png)

Next, we define the [`ImplicitOptimizer`](@ref) and `solve` the problem.

```@example 2

opt = ImplicitOptimizer(4e-1)
g_(x) = x[1] <= 1 ? Inf : norm(x) 
res = solve(prob, basis, opt, normalize = false, denoise = false, maxiter = 1000, g = g_);
println(res) # hide
```

As we can see, the [`DataDrivenSolution`](@ref) already has good metrics. Inspection of the underlying system shows that the original equations have been recovered correctly:

```@example 2
system = result(res); # hide
println(system)
```

!!! warning
    Right now, `Implicit` results cannot be simulated without further processing in `ModelingToolkit`

## Implicit Nonlinear Dynamics : Cartpole

The following is another example on how to use the [`ImplicitOptimizer`](@ref) and is taken from the [original paper](https://royalsocietypublishing.org/doi/10.1098/rspa.2020.0279). 

As always, we start by creating a corresponding dataset.

```@example 5
using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra
using Plots
gr()


function cart_pole(u, p, t)
    du = similar(u)
    F = -0.2 + 0.5*sin(6*t) # the input
    du[1] = u[3]
    du[2] = u[4]
    du[3] = -(19.62*sin(u[1])+sin(u[1])*cos(u[1])*u[3]^2+F*cos(u[1]))/(2-cos(u[1])^2)
    du[4] = -(sin(u[1])*u[3]^2 + 9.81*sin(u[1])*cos(u[1])+F)/(2-cos(u[1])^2)
    return du
end

u0 = [0.3; 0; 1.0; 0]
tspan = (0.0, 5.0)
dt = 0.1
cart_pole_prob = ODEProblem(cart_pole, u0, tspan)
solution = solve(cart_pole_prob, Tsit5(), saveat = dt)

X = solution[:,:]
DX = similar(X)
for (i, xi) in enumerate(eachcol(X))
    DX[:, i] = cart_pole(xi, [], solution.t[i])
end
t = solution.t

ddprob = ContinuousDataDrivenProblem(
    X , t, DX = DX[3:4, :], U = (u,p,t) -> [-0.2 + 0.5*sin(6*t)]
)


plot(solution) # hide
savefig("SINDy_Example_Data_3.png") # hide
```
![](SINDy_Example_Data_3.png)

Next, we define a sufficient [`Basis`](@ref)

```@example 5
@variables u[1:4] du[1:2] x[1:1] t
polys = polynomial_basis(u, 2)
push!(polys, sin.(u[1]))
push!(polys, cos.(u[1]))
push!(polys, sin.(u[1])^2)
push!(polys, cos.(u[1])^2)
push!(polys, sin.(u[1]).*u[3:4]...)
push!(polys, sin.(u[1]).*u[3:4].^2...)
push!(polys, sin.(u[1]).*cos.(u[1])...)
push!(polys, sin.(u[1]).*cos.(u[1]).*u[3:4]...)
push!(polys, sin.(u[1]).*cos.(u[1]).*u[3:4].^2...)
implicits = [du;  du[1] .* u; du[2] .* u; du .* cos(u[1]);   du .* cos(u[1])^2; polys]
push!(implicits, x...)
push!(implicits, x[1]*cos(u[1]))
push!(implicits, x[1]*sin(u[1]))

basis= Basis(implicits, [u; du], controls = x,  iv = t)
```

And solve the problem by varying over a sufficient set of thresholds for the associated optimizer.
Additionally we activate the `scale_coefficients` option for the [`ImplicitOptimizer](@ref), which helps to find sparse equations by normalizing the resulting coefficient matrix after each suboptimization.

To evaluate the pareto optimal solution over, we use the functions `f` and `g` which can be passed as keyworded arguements into the `solve` function. `f` is a function with different signatures for different optimizers, but returns the ``L_0`` norm of the coefficients and the ``L_2`` error of the current model. `g` takes this vector and projects it down onto a scalar, using the [`AIC`](@ref) per default. However, here we want to use the `AIC`  of the output of `f`. A noteworthy exception is of course, that we want only results with two or more active coefficents. Hence we modify `g` accordingly.

```@example 5
λ = [1e-4;5e-4;1e-3;2e-3;3e-3;4e-3;5e-3;6e-3;7e-3;8e-3;9e-3;1e-2;2e-2;3e-2;4e-2;5e-2;
6e-2;7e-2;8e-2;9e-2;1e-1;2e-1;3e-1;4e-1;5e-1;6e-1;7e-1;8e-1;9e-1;1;1.5;2;2.5;3;3.5;4;4.5;5;
6;7;8;9;10;20;30;40;50;100;200];
opt = ImplicitOptimizer(λ)

# Compute the AIC
g(x) = x[1] <= 1 ? Inf : 2*f[1]-2*log(f[2])
res = solve(ddprob, basis, opt, du, maxiter = 1000, g = g, scale_coefficients = true)

println(res)
println(result(res))
println(parameter_map(res))
```