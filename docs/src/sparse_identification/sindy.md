# Sparse Identification of Nonlinear Dynamics

[Sparse Identification of Nonlinear Dynamics](https://www.pnas.org/content/113/15/3932) - or SINDy - identifies the equations of motion of a system as the result of a sparse regression over a chosen basis. In particular, it tries to find coefficients ``\Xi`` such that

```math
\Xi = \min ~ \left\lVert Y^{T} - \Theta(X, p, t)^{T} \Xi \right\rVert_{2} + \lambda ~ \left\lVert \Xi \right\rVert_{1}
```

where in most cases ``Y``is the data matrix containing the derivatives of the state data stored in ``X``. ``\Theta`` is a matrix containing candidate functions ``\xi`` over the measurements in ``X``.



## Example

As in the original paper, we will estimate the [Lorenz System]().
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

To perform the sparse identification on our data, we need to define an `Optimiser`. Here we will use `STRRidge` which is described in the original paper. The threshold of the optimiser is set to `0.1`.

```@example sindy_1
opt = STRRidge(0.1)
Ψ = SInDy(X, DX, basis, maxiter = 100, opt = opt, normalize = true)
print(Ψ, show_parameter = true)
```

The last command prints the equations from the `SInDyResult` with its numerical values. To generate a numerical model usable in `DifferentialEquations`, we simply use the `ODESystem` function from `ModelingToolkit`.

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
```
Finally, lets investigate the error of the chaotic equations:

![](lorenz_error.png)

## Functions

```@docs
SInDy
sparse_regression
```

## Optimiser


where `data` is a matrix of observed values (each column is a timepoint,
each row is an observable), `dx` is the matrix of derivatives of the observables,
`basis` is a `Basis`. This function will return a `Basis` constructed via a
sparse regression over initial `basis`.

`DataDrivenDiffEq` comes with some sparsifying regression algorithms (of the
abstract type `AbstractOptimiser`). Currently these are `STRRidge(threshold)`
from the [original paper](https://www.pnas.org/content/113/15/3932), a custom lasso
implementation via the alternating direction method of multipliers `ADMM(threshold, weight)`
and the `SR3(threshold, relaxation, proxoperator)` for [sparse relaxed regularized regression](https://arxiv.org/pdf/1807.05411.pdf). Here `proxoperator` can be any norm defined
via [ProximalOperators](https://github.com/kul-forbes/ProximalOperators.jl).
The `SInDy` algorithm can be called with all of the above via

```julia
opt = SR3()
dudt = SInDy(data, dx, basis, maxiter = 100, opt = opt)
```

In most cases, `STRRidge` works fine with little iterations (passed in via the `maxiter` argument ).
For larger datasets, `SR3` is in general faster even though it requires more iterations to converge.

Additionally, the boolean arguments `normalize` and `denoise` can be passed, which normalize the data matrix
or reduce it via the [optimal threshold for singular values](http://arxiv.org/abs/1305.5870).
