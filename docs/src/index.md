# DataDrivenDiffEq.jl

**DataDrivenDiffEq.jl** is a package for finding systems of equations of automatically from a dataset.

The methods in this package take in data and return the model which generated the data. A known model is not required as input. These methods can estimate equation-free and equation-based models for discrete, continuous differential equations or direct mappings.

There are two main types of estimation, depending on if you need the result to be human-understandable:
+ Structural identification - returns a human readable result in symbolic form.
+ Structural estimation - returns a function that predicts the derivative and generates a correct time series, but is not necessarily human readable.

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
Pkg.status(;mode = PKGMODE_MANIFEST) # hide
```
```@raw html
</details>
```
```@raw html
You can also download the 
<a href="
```
```@eval
using TOML
version = TOML.parse(read("../../Project.toml",String))["version"]
name = TOML.parse(read("../../Project.toml",String))["name"]
link = "https://github.com/SciML/"*name*".jl/tree/gh-pages/v"*version*"/assets/Manifest.toml"
```
```@raw html
">manifest</a> file and the
<a href="
```
```@eval
using TOML
version = TOML.parse(read("../../Project.toml",String))["version"]
name = TOML.parse(read("../../Project.toml",String))["name"]
link = "https://github.com/SciML/"*name*".jl/tree/gh-pages/v"*version*"/assets/Project.toml"
```
```@raw html
">project</a> file.
```