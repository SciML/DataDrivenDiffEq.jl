# Implicit Sparse Identification of Nonlinear Dynamics

While `SInDy` works well for ODEs, some systems take the form of rational functions `dx = f(x) / g(x)`. These can be inferred via `ISInDy`, which extends `SInDy` [for Implicit problems](https://ieeexplore.ieee.org/abstract/document/7809160).

```julia
dudt = ISInDy(data, dx,basis)
```

The function call returns a `SparseIdentificationResult`. The signature of the additional arguments is equal to `SInDy`, but requires an `AbstractSubspaceOptimser`. Currently `DataDrivenDiffEq` just implements `ADM()` based on [alternating directions](https://arxiv.org/pdf/1412.4659.pdf). `rtol` gets passed into the derivation of the `nullspace` via `LinearAlgebra`.

## Functions

```@docs
ISInDy
DataDrivenDiffEq.Optimise.ADM
```
