# DataDrivenDiffEq.jl

**DataDrivenDiffEq.jl** is a package for finding the governing equations of motion automatically from a dataset.

The methods in this package take in data and return the differential equation model which generated the data. A known model is not required as input. These methods can estimate equation-free and equation-based models for discrete and continuous differential equations.

There are two main types of estimation, depending on if you need the result to be human-understandable:
+ Structural identification - returns a human readable result in symbolic form.
+ Structural estimation - returns a function that predicts the derivative and generates a correct time series, but is not necessarily human readable.

## Package Overview

Currently, the following algorithms for structural estimation and identification are implemented. Please note that all the algorithms have been unified under a single mathematical framework, so the interface might be a little different than what you expect.

+ Dynamic Mode Decomposition (DMD)
+ Extended Dynamic Mode Decomposition
+ Sparse Identification of Nonlinear Dynamics (SINDy)
+ Implicit Sparse Identification of Nonlinear Dynamics


## Installation

To use [DataDrivenDiffEq.jl](https://github.com/SciML/DataDrivenDiffEq.jl), install via:

```julia
using Pkg
Pkg.add("DataDrivenDiffEq")
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