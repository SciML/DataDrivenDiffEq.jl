# Developer API

The interfaces on this page are versioned for packages that implement DataDrivenDiffEq
algorithms. Application users should use the problem, basis, solver, and solution APIs
instead. Code outside solver implementations should not subtype or call these interfaces.

## Core abstractions

```@docs
DataDrivenDiffEq.AbstractBasis
DataDrivenDiffEq.AbstractDataDrivenAlgorithm
DataDrivenDiffEq.AbstractDataDrivenResult
DataDrivenDiffEq.AbstractDataDrivenProblem
DataDrivenDiffEq.ABSTRACT_DIRECT_PROB
DataDrivenDiffEq.ABSTRACT_DISCRETE_PROB
DataDrivenDiffEq.ABSTRACT_CONT_PROB
DataDrivenDiffEq.InternalDataDrivenProblem
```

## Extension hooks

```@docs
DataDrivenDiffEq.get_fit_targets
DataDrivenDiffEq.is_implicit
DataDrivenDiffEq.is_controlled
DataDrivenDiffEq.get_f
DataDrivenDiffEq.get_implicit_data
DataDrivenDiffEq.get_oop_args
DataDrivenDiffEq.remake_problem
DataDrivenDiffEq.assert_lhs
DataDrivenDiffEq.apply_transform
DataDrivenDiffEq.apply_transform!
DataDrivenDiffEq.__construct_basis
```
