# Implicit Sparse Identification of Nonlinear Dynamics

While `SInDy` works well for ODEs, some systems take the form of rational functions `dx = f(x) / g(x)`. These can be inferred via `ISInDy`, which extends `SInDy` [for Implicit problems](https://ieeexplore.ieee.org/abstract/document/7809160). In particular, it solves

```math
\Xi = \min ~ \left\lVert \Theta(X, p, t)^{T} \Xi \right\rVert_{2} + \lambda ~ \left\lVert \Xi \right\rVert_{1}
```

where ``\Xi`` lies in the nullspace of ``\Theta``.

## Example

Let's try to infer the [Michaelis-Menten Kinetics](https://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics), like in the corresponding paper. We start by generating the
corresponding data.

```@example isindy_1
using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra
using Plots
gr()

function michaelis_menten(u, p, t)
    [0.6 - 1.5u[1]/(0.3+u[1])]
end

u0 = [0.5]
tspan = (0.0, 5.0)
problem = ODEProblem(michaelis_menten, u0, tspan)
solution = solve(problem, Tsit5(), saveat = 0.1)
plot(solution) # hide
savefig("isindy_example.png")
```
![](isindy_example.png)

```@example isindy_1
X = solution[:,:]
DX = similar(X)
for (i, xi) in enumerate(eachcol(X))
    DX[:, i] = michaelis_menten(xi, [], 0.0)
end

@variables u
basis= Basis([u^i for i in 0:4], [u])
```

The signature of `ISInDy` is equal to `SInDy`, but requires an `AbstractSubspaceOptimser`. Currently, `DataDrivenDiffEq` just implements `ADM()` based on [alternating directions](https://arxiv.org/pdf/1412.4659.pdf). `rtol` gets passed into the derivation of the `nullspace` via `LinearAlgebra`.


```@example isindy_1
opt = ADM(1e-1)
```

Since `ADM()` returns sparsified columns of the nullspace we need to find a pareto optimal solution. To achieve this, we provide an `AbstractScalarizationMethod` to `ISInDy`. This allows us to evaluate each individual column of the sparse matrix on its 0-norm (sparsity) and the 2-norm of the matrix vector product of ``\Theta^T \xi`` (nullspace). Here, we want to set the focus on the the magnitude of the deviation from the nullspace using `WeightedSum`.

```@example isindy_1

f_target = WeightedSum([0.01 1.0], x->identity(x))

Ψ = ISInDy(X, DX, basis, opt = opt, maxiter = 100, rtol = 0.9, alg = f_target)
nothing #hide
```

The function call returns a `SparseIdentificationResult`.
As in [Sparse Identification of Nonlinear Dynamics](@ref), we can transform the `SparseIdentificationResult` into an `ODESystem`.

```@example isindy_1
# Transform into ODE System
sys = ODESystem(Ψ)
dudt = ODEFunction(sys)
ps = parameters(Ψ)

estimator = ODEProblem(dudt, u0, tspan, ps)
estimation = solve(estimator, Tsit5(), saveat = 0.1)

plot(solution, color = :red, label = "True") # hide
plot!(estimation, color = :green, label = "Estimation") # hide
savefig("isindy_example_final.png") # hide
```
![](isindy_example_final.png)

The model recovered by `ISInDy` is  correct

```@example isindy_1
print_equations(Ψ)
```

The parameters are off a little, but, as before, we can use `DiffEqFlux` to tune them.

## Functions

```@docs
ISInDy
DataDrivenDiffEq.Optimize.ADM
```
