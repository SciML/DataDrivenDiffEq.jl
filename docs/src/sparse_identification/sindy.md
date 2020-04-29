# Sparse Identification of Nonlinear Dynamics

[Sparse Identification of Nonlinear Dynamics](https://www.pnas.org/content/113/15/3932) - or SINDy - identifies the equations of motion of a system as the result of a sparse regression over a chosen basis. In particular, it tries to find coefficients ``\Xi`` such that

```math
\Xi = \min ~ \left\lVert Y^{T} - \Theta(X, p, t)^{T} \Xi \right\rVert_{2} + \lambda ~ \left\lVert \Xi \right\rVert_{1}
```

where in most cases ``Y``is the data matrix containing the derivatives of the state data stored in ``X``. ``\Theta`` is a matrix containing candidate functions ``\xi`` over the measurements in ``X``.



## Example

As in the original paper, we will estimate the [Lorenz System](https://en.wikipedia.org/wiki/Lorenz_system).
First, lets create the necessary data and have a look at the trajectory.

```@example sindy_1
using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq

using LinearAlgebra
using Plots
gr()

# Create a test problem
function lorenz(u,p,t)
    x, y, z = u
    ẋ = 10.0*(y - x)
    ẏ = x*(28.0-z) - y
    ż = x*y - (8/3)*z
    return [ẋ, ẏ, ż]
end

u0 = [-8.0; 7.0; 27.0]
p = [10.0; -10.0; 28.0; -1.0; -1.0; 1.0; -8/3]
tspan = (0.0,100.0)
dt = 0.001
problem = ODEProblem(lorenz,u0,tspan)
solution = solve(problem, Tsit5(), saveat = dt, atol = 1e-7, rtol = 1e-8)

plot(solution,vars=(1,2,3), legend = false) #hide
savefig("lorenz.png") #hide
```
![](lorenz.png)

Additionally, we generate the *ideal* derivative data.

```@example sindy_1
X = Array(solution)
DX = similar(X)
for (i, xi) in enumerate(eachcol(X))
    DX[:,i] = lorenz(xi, [], 0.0)
end
```

To generate the symbolic equations, we need to define a ` Basis` over the variables `x y z`. In this example, we will use all monomials up to degree of 4 and their products:

```@example sindy_1
@variables x y z
u = Operation[x; y; z]
polys = Operation[]
for i ∈ 0:4
    for j ∈ 0:i
        for k ∈ 0:j
            push!(polys, u[1]^i*u[2]^j*u[3]^k)
            push!(polys, u[2]^i*u[3]^j*u[1]^k)
            push!(polys, u[3]^i*u[1]^j*u[2]^k)
        end
    end
end

basis = Basis(polys, u)
nothing #hide
```

*A `Basis` consists of unique functions, so duplicates will be included just once*

To perform the sparse identification on our data, we need to define an `Optimiser`. Here we will use `STRRidge` which is described in the original paper. The threshold of the optimiser is set to `0.1`. An overview of the different optimisers can be found below.

```@example sindy_1
opt = STRRidge(0.1)
Ψ = SInDy(X, DX, basis, maxiter = 100, opt = opt, normalize = true)
```

`Ψ` is a `SInDyResult`, which stores some about the regression. As we can see, we have 7 active terms inside the model.
To look at the equations, simply type

```@example sindy_1
print_equations(Ψ)
```

First, lets have a look at the ``L2``-Error and Akaikes Information Criterion of the result

```@example sindy_1
get_error(Ψ)
```

```@example sindy_1
get_aicc(Ψ)
```

We can also access the coefficient matrix ``\Xi`` directly via `get_coefficients(Ψ)`.

To generate a numerical model usable in `DifferentialEquations`, we simply use the `ODESystem` function from `ModelingToolkit`.
The resulting parameters used for the identification can be accessed via `parameters(Ψ)`. The returned vector also includes the parameters of the original `Basis` used to generate the result.

```@example sindy_1
ps = parameters(Ψ)
sys = ODESystem(Ψ)
dudt = ODEFunction(sys)

prob = ODEProblem(dudt, u0, tspan, ps)
sol = solve(prob, Tsit5(), saveat = solution.t, atol = 1e-7, rtol = 1e-8)

ϵ = norm.(eachcol(solution .- sol)) # hide
plot(solution.t, ϵ .+ eps(), yaxis = :log, legend = false) # hide
xlabel!("Time [s]") # hide
ylabel!("L2 Error") # hide
savefig("lorenz_error.png") # hide
plot(solution, vars = (0, 1), label = "True") # hide
plot!(sol, vars = (0,1), label = "Estimation") # hide
savefig("lorenz_trajectory_estimate.png") # hide
```

Lets have a look at the trajectory of ``u_{1}(t)``.

![](lorenz_trajectory_estimate.png)

Finally, lets investigate the error of the chaotic equations:

![](lorenz_error.png)

Which resembles the papers results. Next, we could use [classical parameter estimation methods](https://docs.sciml.ai/stable/analysis/parameter_estimation/) or use [DiffEqFlux](https://github.com/SciML/DiffEqFlux.jl) to fine tune our result (if needed ).

## Functions

```@docs
SInDy
sparse_regression
```

## Optimiser

`DataDrivenDiffEq` comes with some implementations for sparse regression included. All of these are stored inside the
`DataDrivenDiffEq.Optimise` package and extend the `AbstractOptimiser`.



```@docs
STRRidge
ADMM
SR3
```
