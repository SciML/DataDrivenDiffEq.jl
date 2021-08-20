# Symbolic Regression

`DataDrivenDiffEq` includes the following symbolic regression algorithms.

## SymbolicRegression

!!! warning
    This feature requires the explicit loading of [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl) in addition to `DataDrivenDiffEq`. It will _only_ be useable if loaded like:
    ```julia
    using DataDrivenDiffEq
    using SymbolicRegression
    ```

### Symbolic Regression
See the [tutorial](@ref symbolic_regression_tutorial)

```@docs
EQSearch
```

## OccamNet
See the [tutorial](@ref occam_net_tutorial)

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
