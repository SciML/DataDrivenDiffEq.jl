```@meta
EditURL = "<unknown>/docs/src/libs/datadrivensparse/example_02.jl"
```

# [Sparse Identification with noisy data](@id noisy_sindy)

Many real world data sources are corrupted with measurment noise, which can have
a big impact on the recovery of the underlying equations of motion. This example show how we can
use [`collocation`](@ref collocation) and [`batching`](@ref DataSampler) to perform SINDy in the presence of
noise.

````@example example_02
using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenSparse
using StableRNGs
using Plots
gr()

rng = StableRNG(42)

function pendulum(u, p, t)
    x = u[2]
    y = -9.81sin(u[1]) - 0.3u[2]^3 -3.0*cos(u[1]) - 10.0*exp(-((t-5.0)/5.0)^2)
    return [x;y]
end

u0 = [0.99π; -1.0]
tspan = (0.0, 15.0)
prob = ODEProblem(pendulum, u0, tspan)
sol = solve(prob, Tsit5(), saveat = 0.01);
nothing #hide
````

We add random noise to our measurements.

````@example example_02
X = sol[:,:] + 0.2 .* randn(rng, size(sol));
ts = sol.t;

plot(ts, X', color = :red)
plot!(sol, color = :black)
````

To estimate the system, we first create a [`DataDrivenProblem`](@ref) via feeding in the measurement data.
Using a [Collocation](@ref) method, it automatically provides the derivative and smoothes the trajectory. Control signals can be passed
in as a function `(u,p,t)->control` or an array of measurements.

````@example example_02
prob = ContinuousDataDrivenProblem(X, ts, GaussianKernel() ,
    U = (u,p,t)->[exp(-((t-5.0)/5.0)^2)], p = ones(2))

plot(prob, size = (600,600))
````

Now we infer the system structure. First we define a [`Basis`](@ref) which collects all possible candidate terms.
Since we want to use SINDy, we call `solve` with an [`Optimizer`](@ref sparse_optimization), in this case [`STLSQ`](@ref) which iterates different sparsity thresholds
and returns a pareto optimal solution of the underlying [`sparse_regression!`](@ref). Note that we include the control signal in the basis as an additional variable `c`.

````@example example_02
@variables u[1:2] c[1:1]
@parameters w[1:2]
u = collect(u)
c = collect(c)
w = collect(w)

h = Num[sin.(w[1].*u[1]);cos.(w[2].*u[1]); polynomial_basis(u, 5); c]

basis = Basis(h, u, parameters = w, controls = c);
println(basis) # hide
````

To solve the problem, we also define a [`DataSampler`](@ref) which defines randomly shuffled minibatches of our data and selects the
best fit.

````@example example_02
sampler = DataProcessing(split = 0.8, shuffle = true, batchsize = 30, rng = rng)
λs = exp10.(-10:0.1:0)
opt = STLSQ(λs)
res = solve(prob, basis, opt, options = DataDrivenCommonOptions(data_processing = sampler, digits = 1))
````

!!! info
    A more detailed description of the result can be printed via `print(res, Val{true})`, which also includes the discovered equations and parameter values.

Where the resulting [`DataDrivenSolution`](@ref) stores information about the inferred model and the parameters:

````@example example_02
system = get_basis(res)
params = get_parameter_map(system)
println(system) #hide
println(params) #hide
````

And a visual check of the result can be perfomed via plotting the result

````@example example_02
plot(
    plot(prob), plot(res), layout = (1,2)
)
````

## [Copy-Pasteable Code](@id autoregulation_copy_paste)

```julia
using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenSparse
using StableRNGs

rng = StableRNG(42)

function pendulum(u, p, t)
    x = u[2]
    y = -9.81sin(u[1]) - 0.3u[2]^3 -3.0*cos(u[1]) - 10.0*exp(-((t-5.0)/5.0)^2)
    return [x;y]
end

u0 = [0.99π; -1.0]
tspan = (0.0, 15.0)
prob = ODEProblem(pendulum, u0, tspan)
sol = solve(prob, Tsit5(), saveat = 0.01);

X = sol[:,:] + 0.2 .* randn(rng, size(sol));
ts = sol.t;

prob = ContinuousDataDrivenProblem(X, ts, GaussianKernel() ,
    U = (u,p,t)->[exp(-((t-5.0)/5.0)^2)], p = ones(2))

@variables u[1:2] c[1:1]
@parameters w[1:2]
u = collect(u)
c = collect(c)
w = collect(w)

h = Num[sin.(w[1].*u[1]);cos.(w[2].*u[1]); polynomial_basis(u, 5); c]

basis = Basis(h, u, parameters = w, controls = c);
println(basis) # hide

sampler = DataProcessing(split = 0.8, shuffle = true, batchsize = 30, rng = rng)
λs = exp10.(-10:0.1:0)
opt = STLSQ(λs)
res = solve(prob, basis, opt, options = DataDrivenCommonOptions(data_processing = sampler, digits = 1))

system = get_basis(res)
params = get_parameter_map(system)
println(system) #hide
println(params) #hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

