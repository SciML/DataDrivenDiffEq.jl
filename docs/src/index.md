# DataDrivenDiffEq.jl

DataDrivenDiffEq.jl is a package for finding systems of equations automatically from a dataset.

The methods in this package take in data and return the model which generated the data. A known model is not required as input. These methods can estimate equation-free and equation-based models for discrete, continuous differential equations or direct mappings.

There are two main types of estimation, depending on if you need the result to be human-understandable:

  - Structural identification - returns a human-readable result in symbolic form.
  - Structural estimation - returns a function that predicts the derivative and generates a correct time series, but is not necessarily human-readable.

A quick-start example:

```@example quickstart
using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using DataDrivenSparse
using LinearAlgebra

# Create a test problem
function lorenz(u, p, t)
    x, y, z = u

    ẋ = 10.0 * (y - x)
    ẏ = x * (28.0 - z) - y
    ż = x * y - (8 / 3) * z
    return [ẋ, ẏ, ż]
end

u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 100.0)
dt = 0.1
prob = ODEProblem(lorenz, u0, tspan)
sol = solve(prob, Tsit5(), saveat = dt)

## Start the automatic discovery
ddprob = DataDrivenProblem(sol)

@variables t x(t) y(t) z(t)
u = [x; y; z]
basis = Basis(polynomial_basis(u, 5), u, iv = t)
opt = STLSQ(exp10.(-5:0.1:-1))
ddsol = solve(ddprob, basis, opt, options = DataDrivenCommonOptions(digits = 1))
println(get_basis(ddsol))
```

## Installation

To use [DataDrivenDiffEq.jl](https://github.com/SciML/DataDrivenDiffEq.jl), install via:

```julia
using Pkg
Pkg.add("DataDrivenDiffEq")
```

## Package Overview

Several algorithms for structural estimation and identification are implemented in the following subpackages.

### Koopman Based Inference

Uses Dynamic Mode Decomposition (DMD) and Extended Dynamic Mode Decomposition (EDMD) on discrete and continuous differential equations to infer an approximation of the corresponding Koopman operator (discrete case) or generator (continuous case).

To use this functionality, install [DataDrivenDMD](@ref) via:

```julia
using Pkg
Pkg.add("DataDrivenDMD")
```

### Sparse Regression

Uses Sparse Regression algorithms to find a suitable and sparse combination of basis functions to approximate a system of (differential) equations.

To use this functionality, install [DataDrivenSparse](@ref) via:

```julia
using Pkg
Pkg.add("DataDrivenSparse")
```

### Symbolic Regression

Uses SymbolicRegression.jl to find a suitable set of equations to match the data.

To use this functionality, install [DataDrivenSR](@ref) via:

```julia
using Pkg
Pkg.add("DataDrivenSR")
```

## Contributing

  - Please refer to the
    [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac)
    for guidance on PRs, issues, and other matters relating to contributing to SciML.

  - See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
  - There are a few community forums:
    
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Slack](https://julialang.org/slack/),
        you can message @AlCap23 to start a discussion.
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
      + On the [Julia Discourse forums](https://discourse.julialang.org)
      + See also [SciML Community page](https://sciml.ai/community/)

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
