# Sparse Identification of Nonlinear Dynamics

[Sparse Identification of Nonlinear Dynamics](https://www.pnas.org/content/113/15/3932) - or SINDy - identifies the equations of motion of a system as the result of a sparse regression over a chosen basis. In particular, it tries to find coefficients ``\Xi`` such that

```math
\Xi = \min ~ \left\lVert Y^{T} - \Theta(X, p, t)^{T} \Xi \right\rVert_{2} + \lambda ~ \left\lVert \Xi \right\rVert_{1}
```

where in most cases ``Y``is the data matrix containing the derivatives of the state data stored in ``X``. ``\Theta`` is a matrix containing candidate functions ``\xi`` over the measurements in ``X``.



## Example

As in the original paper, we will estimate the [Lorenz System]()
```@example sindy_1
using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra
using Plots
gr()

function lorenz(u,p,t)
    x, y, z = u

    ẋ = 10.0*(y - x)
    ẏ = x*(28.0-z) - y
    ż = x*y - (8/3)*z
    return [ẋ, ẏ, ż]
end

u0 = [1.0;0.0;0.0]
tspan = (0.0,100.0)
dt = 0.005

problem = ODEProblem(lorenz,u0,tspan)
solution = solve(problem, Tsit5(), saveat = dt)

plot(sol,vars=(1,2,3))
```

```@example sindy_1
X = Array(solution)
DX = solution(solution.t, Val{1})

```

```@example sindy_1
@variables u[1:3]

polys = [u[1]^0]
for i ∈ 0:3
    for j ∈ 0:3
        for  k ∈ 0:3
            push!(polys, u[1]^i * u[2]^j * u[3]^k)
        end
    end
end

h = [1u[1];1u[2]; cos(u[1]); sin(u[1]); u[1]*u[2]; u[1]*sin(u[2]); u[2]*cos(u[2]); polys...]
basis = Basis(h, u)
```

```@example sindy_1
opt = STRRidge(0.1)
Ψ = SInDy(X, DX, basis, maxiter = 100, opt = opt, normalize = false)
```

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
