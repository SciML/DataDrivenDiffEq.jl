## Basis

Many of the methods require the definition of a `Basis` on observables or
functional forms. A `Basis` is generated via:

```julia
Basis(h, u, parameters = [], iv = nothing)
```

where `h` is either a vector of ModelingToolkit `Operation`s for the valid functional
forms or a general function with the typical DiffEq signature `h(u,p,t)` which can be used with an  `Operation` or vector of `Operation`. `u` are the ModelingToolkit `Variable`s used to describe the Basis, and
`parameters` are the optional ModelingToolkit `Variable`s used to describe the
parameters in the basis elements. `iv` represents the independent variable of the system, the time.

```@docs
Basis
```

## Functions

```@docs
parameters
variables
DataDrivenDiffEq.independent_variable
jacobian
dynamics
```

## Adaptation
```@docs
push!
deleteat!
merge
merge!
```
