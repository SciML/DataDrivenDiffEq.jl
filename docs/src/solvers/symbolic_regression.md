# [Symbolic Regression](@id symbolic_regression_api)

`DataDrivenDiffEq` includes the following symbolic regression algorithms.

## [EQSearch](@id eqsearch_api)

!!! warning
    This feature requires the explicit loading of [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl) in addition to `DataDrivenDiffEq`. It will _only_ be useable if loaded like:
    ```julia
    using DataDrivenDiffEq
    using SymbolicRegression
    ```
    Currently `DataDrivenDiffEq` supports version 0.6.14 up to 0.6.19.

This algorithm wraps [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl).
### Symbolic Regression
See the [tutorial](@ref symbolic_regression_simple).

```@docs
EQSearch
```

## [OccamNet](@id occamnet_api)

!!! warning
    This feature requires the explicit loading of [Flux.jl](https://fluxml.ai/) in addition to `DataDrivenDiffEq`. It will _only_ be useable if loaded like:
    ```julia
    using DataDrivenDiffEq
    using Flux
    ```

```@docs
OccamNet
OccamSR
ProbabilityLayer
```
### Related Functions

```@docs
set_temp!
probability
logprobability
probabilities
logprobabilities
```
