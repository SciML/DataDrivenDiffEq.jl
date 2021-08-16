# DataDrivenDiffEq.jl

[DataDrivenDiffEq.jl](https://github.com/SciML/DataDrivenDiffEq.jl) is a package for finding the governing equations of motion automatically from a dataset.

The methods in this package take in data and return the differential equation model which generated the data. A known model is not required as input. These methods can estimate equation-free and equation-based models for discrete and continous differential equations.

There are two main types of estimation, depending on if you need the result to be human-understandable:
+ Structural identification - returns a human readable result in symbolic form
+ Structural estimation - returns a function that predicts the derivative and gnerates a correct time series, but is not necessarily human readable.

## Package Overview

[DataDrivenDiffEq.jl](https://github.com/SciML/DataDrivenDiffEq.jl) currently implements the following algorithms for structural estimation. Please note that all the algorithms have been unified under a single mathematical framework, so that interface might be a little different than what you expect.

+ Dynamic Mode Decomposition (DMD)
+ Extended Dynamic Mode Decomposition
+ Sparse Identification of Nonlinear Dynamics (SINDy)
+ Implicit Sparse Identification of Nonlinear Dynamics


## Installation

To use [DataDrivenDiffEq.jl](https://github.com/SciML/DataDrivenDiffEq.jl), simply install it via:

```julia
]add DataDrivenDiffEq
using DataDrivenDiffEq
```

## Basic Usage

All algorithms have been unified under a common mathematical interface. TODO