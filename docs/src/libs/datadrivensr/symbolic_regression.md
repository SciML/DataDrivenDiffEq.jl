# DataDrivenSR

DataDrivenSR provides an API to [`SymbolicRegression.jl`](https://github.com/MilesCranmer/SymbolicRegression.jl) to infer arbitrary systems of equations.

```math
y_{i} = f(x_{i}, p, t_i, u_{i})
```

For examples see the tutorial section.

## [Algorithms](@id sr_algorithms)

```@docs
EQSearch
```

## Reexported API

`using DataDrivenSR` also brings two SymbolicRegression names into scope, so that an
`EQSearch` can be configured without importing SymbolicRegression separately:

  - `SymbolicRegression` — the module itself, for qualified access. This is how the
    examples on this site build their options:
    `SymbolicRegression.Options(binary_operators = [+, *], ...)`.
  - `Options` — SymbolicRegression's option type, the value of the `eq_options` field
    of [`EQSearch`](@ref).

**DataDrivenSR does not own or document either name.** Both are owned and documented by
[SymbolicRegression.jl](https://ai.damtp.cam.ac.uk/symbolicregression/dev/); see its
API reference for what `Options` accepts.

Nothing else from SymbolicRegression is reexported. `equation_search`, `Node`,
`Population`, the operator and template machinery, and the `LossFunctions` types
(`L1DistLoss` and friends) reexported by SymbolicRegression must be imported from
SymbolicRegression or LossFunctions directly.
