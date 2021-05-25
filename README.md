# DataDrivenDiffEq.jl

[![Build Status](https://github.com/SciML/DataDrivenDiffEq.jl/workflows/CI/badge.svg)](https://github.com/SciML/DataDrivenDiffEq.jl/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/JuliaDiffEq/DataDrivenDiffEq.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaDiffEq/DataDrivenDiffEq.jl?branch=master)
[![codecov.io](http://codecov.io/github/JuliaDiffEq/DataDrivenDiffEq.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaDiffEq/DataDrivenDiffEq.jl?branch=master)
[![DOI](https://zenodo.org/badge/212827023.svg)](https://zenodo.org/badge/latestdoi/212827023)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)


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

# Create a test problem
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
prob = ODEProblem(lorenz,u0,tspan)
sol = solve(prob, Tsit5(), saveat = dt, progress = true)

# Differential data from equations
X = Array(sol)
DX = similar(X)
for (i, xi) in enumerate(eachcol(X))
    DX[:,i] = lorenz(xi, [], 0.0)
end

## Start the automatic discovery
ddprob = ContinuousDataDrivenProblem(X, sol.t, DX = DX)

@variables t x(t) y(t) z(t)
u = [x;y;z]
basis = Basis(polynomial_basis(u, 5), u, iv = t)
opt = STLSQ(exp10.(-5:0.1:-1))
ddsol = solve(ddprob, basis, opt, normalize = true)
system = result(ddsol)
```

```
Model ##Basis#350 with 3 equations
x(t) y(t) z(t)
Parameters : 7
Independent variable: t
Equations
Differential(t)(x(t)) = p₁*x(t) + p₂*y(t)
Differential(t)(y(t)) = p₃*x(t) + p₄*y(t) + p₅*x(t)*z(t)
Differential(t)(z(t)) = p₇*z(t) + p₆*x(t)*y(t)
```
