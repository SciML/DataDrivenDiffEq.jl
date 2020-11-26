# Implicit Sparse Identification of Nonlinear Dynamics

While `SINDy` works well for ODEs, some systems take the form of [DAE](https://diffeq.sciml.ai/stable/types/dae_types/)s. A common form is `f(x, p, t) - g(x, p, t)*dx = 0`. These can be inferred via `ISINDy`, which extends `SINDy` [for Implicit problems](https://ieeexplore.ieee.org/abstract/document/7809160). In particular, it solves

```math
\Xi = \min ~ \left\lVert \Theta(X, p, t)^{T} \Xi \right\rVert_{2} + \lambda ~ \left\lVert \Xi \right\rVert_{1}
```

where ``\Xi`` lies in the nullspace of ``\Theta``.

## Example : Michaelis-Menten Kinetics

Let's try to infer the [Michaelis-Menten Kinetics](https://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics), like in the corresponding paper. We start by generating the
corresponding data.

```@example iSINDy_1
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

solution = solve(problem, Tsit5(), saveat = 0.1, atol = 1e-7, rtol = 1e-7)
    
plot(solution) # hide
savefig("iSINDy_example.png")
```
![](iSINDy_example.png)

```@example iSINDy_1
X = solution[:,:]
DX = similar(X)
for (i, xi) in enumerate(eachcol(X))
    DX[:, i] = michaelis_menten(xi, [], 0.0)
end

@variables u
basis= Basis([u^i for i in 0:4], [u])
```

The signature of `ISINDy` is equal to `SINDy`, but requires an `AbstractSubspaceOptimizer`. Currently, `DataDrivenDiffEq` just implements `ADM()` based on [alternating directions](https://arxiv.org/pdf/1412.4659.pdf). `rtol` gets passed into the derivation of the `nullspace` via `LinearAlgebra`.


```@example iSINDy_1
opt = ADM(1.1e-1)
```

Since `ADM()` returns sparsified columns of the nullspace we need to find a pareto optimal solution. To achieve this, we provide a sufficient cost function `g` to `ISINDy`. This allows us to evaluate each individual column of the sparse matrix on its 0-norm (sparsity) and the 2-norm of the matrix vector product of ``\Theta^T \xi`` (nullspace). This is a default setting which can be changed by providing a function `f` which maps the coefficients and the library onto a feature space. Here, we want to set the focus on the the magnitude of the deviation from the nullspace.

```@example iSINDy_1
Ψ = ISINDy(X, DX, basis, opt, g = x->norm([1e-1*x[1]; x[2]]), maxiter = 100)
nothing #hide
```

The function call returns a `SparseIdentificationResult`.
As in [Sparse Identification of Nonlinear Dynamics](@ref), we can transform the `SparseIdentificationResult` into an `ODESystem`.

```@example iSINDy_1
# Transform into ODE System
sys = ODESystem(Ψ)
dudt = ODEFunction(sys)
ps = parameters(Ψ)

estimator = ODEProblem(dudt, u0, tspan, ps)
estimation = solve(estimator, Tsit5(), saveat = 0.1)

plot(solution, color = :red, label = "True") # hide
plot!(estimation, color = :green, label = "Estimation") # hide
savefig("iSINDy_example_final.png") # hide
```
![](iSINDy_example_final.png)

The model recovered by `ISINDy` is  correct

```@example iSINDy_1
print_equations(Ψ)
```

The parameters are off a little, but, as before, we can use `DiffEqFlux` to tune them.


## Example : Cart-Pole with Time-Dependent Control

Implicit dynamics can also be reformulated as an explicit problem as stated in [this paper](https://arxiv.org/pdf/2004.02322.pdf). The algorithm searches the correct equations by trying out all candidate functions as a right hand side and performing a sparse regression onto the remaining set of candidates. Let's start by defining the problem and generate the data:

```@example iSINDy_2

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
tspan = (0.0, 16.0)
dt = 0.001
cart_pole_prob = ODEProblem(cart_pole, u0, tspan)
solution = solve(cart_pole_prob, Tsit5(), saveat = dt)

# Create the differential data
X = solution[:,:]
DX = similar(X)
for (i, xi) in enumerate(eachcol(X))
    DX[:, i] = cart_pole(xi, [], solution.t[i])
end

plot(solution) # hide
savefig("iSINDy_cartpole_data.png") # hide

```
![](iSINDy_cartpole_data.png)

We see that we include a forcing term `F` inside the model which is depending on `t`.
As before, we will also need a `Basis` to derive our equations from:

```@example iSINDy_2
@variables u[1:4] t
polys = Any[]
for i ∈ 0:4
    if i == 0
        push!(polys, u[1]^0)
    else
        if i < 2
            push!(polys, u.^i...)
        else
            push!(polys, u[3:4].^i...)
        end
        
    end
end
push!(polys, sin.(u[1])...)
push!(polys, cos.(u[1]))
push!(polys, sin.(u[1]).*u[3:4]...)
push!(polys, sin.(u[1]).*u[3:4].^2...)
push!(polys, cos.(u[1]).^2...)
push!(polys, sin.(u[1]).*cos.(u[1])...)
push!(polys, sin.(u[1]).*cos.(u[1]).*u[3:4]...)
push!(polys, sin.(u[1]).*cos.(u[1]).*u[3:4].^2...)
push!(polys, -0.2+0.5*sin(6*t))
push!(polys, (-0.2+0.5*sin(6*t))*cos(u[1]))
push!(polys, (-0.2+0.5*sin(6*t))*sin(u[1]))
basis= Basis(polys, u, iv = t)
```

We added the time dependent input directly into the basis to account for its influence.

*NOTE : Including input signals may change in future releases!*

Like for a `SINDy`, we can use any `AbstractOptimizer` with a pareto front optimization over different thresholds. 

```@example iSINDy_2
λ = exp10.(-4:0.1:-1)
g(x) = norm([1e-3; 10.0] .* x, 2)
Ψ = ISINDy(X[:,:], DX[:, :], basis, λ, STRRidge(), maxiter = 100, normalize = false, t = solution.t, g = g)

# Transform into ODE System
sys = ODESystem(Ψ)
dudt = ODEFunction(sys)
ps = parameters(Ψ)

# Simulate
estimator = ODEProblem(dudt, u0, tspan, ps)
sol_ = solve(estimator, Tsit5(), saveat = dt)

plot(solution.t[:], solution[:,:]', color = :red, label = nothing) # hide
plot!(sol_.t, sol_[:, :]', color = :green, label = "Estimation") # hide
savefig("iSINDy_cartpole_estimation.png") # hide
```
![](iSINDy_cartpole_estimation.png)

Let's have a look at the equations recovered. They match up.

```@example iSINDy_2
print_equations(Ψ)
```

Alternatively, we can also use the input as an extended state `x`.

```@example iSINDy_2
@variables u[1:4] t x
polys = Any[]
# Lots of basis functions -> sindy pi can handle more than ADM()
for i ∈ 0:4
    if i == 0
        push!(polys, u[1]^0)
    else
        if i < 2
            push!(polys, u.^i...)
        else
            push!(polys, u[3:4].^i...)
        end
        
    end
end
push!(polys, sin.(u[1])...)
push!(polys, cos.(u[1]))
push!(polys, sin.(u[1]).*u[3:4]...)
push!(polys, sin.(u[1]).*u[3:4].^2...)
push!(polys, cos.(u[1]).^2...)
push!(polys, sin.(u[1]).*cos.(u[1])...)
push!(polys, sin.(u[1]).*cos.(u[1]).*u[3:4]...)
push!(polys, sin.(u[1]).*cos.(u[1]).*u[3:4].^2...)
push!(polys, x)
push!(polys, x*cos(u[1]))
push!(polys, x*sin(u[1]))
basis= Basis(polys, vcat(u, x), iv = t)
```

Now we include the input signal into the extended state array `Xᵤ` and perform a sparse regression.

```@example iSINDy_2
U = -0.2 .+ 0.5*sin.(6*solution.t)
Xᵤ = vcat(X, U')

λ = exp10.(-4:0.5:-1)
g(x) = norm([1e-3; 10.0] .* x, 2)
Ψ = ISINDy(Xᵤ[:,:], DX[:, :], basis, λ, STRRidge(), maxiter = 100, normalize = false, t = solution.t, g = g)
print_equations(Ψ, show_parameter = true)
```

Currently, we *can not* generate an `ODESystem` out of the resulting equations, which is a work in progress.

## Functions

```@docs
ISINDy
```
