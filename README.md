# DataDrivenDiffEq.jl

[![Build Status](https://github.com/SciML/DataDrivenDiffEq.jl/workflows/CI/badge.svg)](https://github.com/SciML/DataDrivenDiffEq.jl/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/JuliaDiffEq/DataDrivenDiffEq.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaDiffEq/DataDrivenDiffEq.jl?branch=master)
[![codecov.io](http://codecov.io/github/JuliaDiffEq/DataDrivenDiffEq.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaDiffEq/DataDrivenDiffEq.jl?branch=master)
[![DOI](https://zenodo.org/badge/212827023.svg)](https://zenodo.org/badge/latestdoi/212827023)


DataDrivenDiffEq.jl is a package in the SciML ecosystem for data-driven differential equation
structural estimation and identification. These tools include automatically discovering equations
from data and using this to simulate perturbed dynamics.

For information on using the package,
[see the stable documentation](https://datadriven.sciml.ai/stable/). Use the
[in-development documentation](https://datadriven.sciml.ai/dev/) for the version of
the documentation which contains the un-released features.

## Quick Demonstration

```julia
## Generate some data by solving a differential equation
########################################################

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

X = Array(solution)
DX = similar(X)
for (i, xi) in enumerate(eachcol(X))
    DX[:,i] = lorenz(xi, [], 0.0)
end

## Now automatically discover the system that generated the data
################################################################

@variables x y z
u = [x; y; z]
polys = Any[]
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

opt = STRRidge(0.1)
Ψ = SINDy(X, DX, basis, opt, maxiter = 100, normalize = true)
print_equations(Ψ)
get_error(Ψ)
```

```
3-dimensional basis in ["x", "y", "z"]
dx = p₁ * x + p₂ * y
dy = p₃ * x + p₄ * y + z * x * p₅
dz = p₆ * z + x * y * p₇

# Error
3-element Array{Float64,1}:
 6.7202639134663155e-12
 3.505423292198665e-11
 1.2876598297504605e-11
```
