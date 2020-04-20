## Basis

Almost all methods require setting some form of a basis on observables or
functional forms. A `Basis` is generated via:

```julia
Basis(h, u, parameters = [])
```

where `h` is a vector of ModelingToolkit `Operation`s for the valid functional
forms, `u` are the ModelingToolkit `Variable`s used to describe the Basis, and
`parameters` are the optional ModelingToolkit `Variable`s used to describe the
parameters in the basis elements.
