# Reexported API

`using DataDrivenDiffEq` brings a fixed set of upstream names into scope, so that the
package's own workflow — write a [`Basis`](basis.md), build a
[problem](problems.md), pick a collocation method, normalize and batch the data,
`solve`, then inspect the [solution](solutions.md) — runs without importing five more
packages first.

**DataDrivenDiffEq does not own or document any of the names on this page.** It only
puts them in scope. Each group names the package that owns them and links to the
documentation you should actually read.

## Symbolic DSL — owned by Symbolics.jl and ModelingToolkitBase.jl

A [`Basis`](basis.md) is written in symbolic variables, so the declaration macros and
the symbolic types come along with the package:

  - `@variables`, `@parameters` — declare the states, controls and parameters a basis
    is written over.
  - `Differential` — the derivative operator used for implicit bases and for the
    equations of a recovered continuous system.
  - `Num`, `Equation` — the symbolic wrapper and equation types a basis is built from
    and returns.
  - `build_function`, `get_variables` — turn symbolic expressions into callable code
    and extract the variables appearing in an expression.

`@variables`, `Differential`, `Num`, `Equation`, `build_function` and `get_variables`
are owned and documented by [Symbolics](https://docs.sciml.ai/Symbolics/stable/);
`@parameters` is owned by
[ModelingToolkit](https://docs.sciml.ai/ModelingToolkit/stable/) (through
ModelingToolkitBase).

Anything else from the symbolic stack — `@register_symbolic`, `substitute`,
`simplify`, `expand_derivatives`, the rewriting machinery — must be imported from
Symbolics or ModelingToolkit directly.

## System accessors — owned by ModelingToolkitBase.jl

A `Basis` is an `AbstractSystem`, so it is inspected with the standard
ModelingToolkit accessors rather than DataDrivenDiffEq-specific ones. These are the
accessors named in the `AbstractBasis` interface (see [Developer
API](developer_api.md)):

  - `equations` — the basis' symbolic equations.
  - `unknowns`, `parameters` — its states and parameters.
  - `observed`, `get_observed` — its observed equations.
  - `independent_variable`, `get_iv` — its independent variable.

Owned and documented by
[ModelingToolkit](https://docs.sciml.ai/ModelingToolkit/stable/). DataDrivenDiffEq
adds methods for its own types; everything else in the `AbstractSystem` interface must
be imported from ModelingToolkit directly.

## Statistical interface — owned by StatsAPI.jl and StatsBase.jl

[`DataDrivenSolution`](solutions.md) and the algorithm result types are
`StatsAPI.StatisticalModel`s, so the quality of a recovered model is read off with the
standard statistical accessors:

  - Information criteria: `aic`, `aicc`, `bic`
  - Fit quality: `rss`, `r2`, `loglikelihood`, `nullloglikelihood`
  - Model size: `dof`, `nobs`
  - Summary: `summarystats`

`aic`, `aicc`, `bic`, `rss`, `r2`, `loglikelihood`, `nullloglikelihood`, `dof` and
`nobs` are owned by [StatsAPI](https://github.com/JuliaStats/StatsAPI.jl);
`summarystats` is owned by
[StatsBase](https://juliastats.org/StatsBase.jl/stable/). The methods
DataDrivenDiffEq defines for them are listed under
[Solutions](solutions.md#Statistical-interface).

## Collocation methods — owned by DataInterpolations.jl

A `ContinuousDataDrivenProblem` can derive its time derivatives by interpolation. Any
of these can be wrapped by [`InterpolationMethod`](utils.md) and passed as the
`collocation` keyword:

  - `LinearInterpolation` (the default), `ConstantInterpolation`,
    `QuadraticInterpolation`, `LagrangeInterpolation`
  - `QuadraticSpline`, `CubicSpline`, `BSplineInterpolation`, `BSplineApprox`
  - `Curvefit`

Owned and documented by
[DataInterpolations](https://docs.sciml.ai/DataInterpolations/stable/). The rest of
the DataInterpolations surface — the remaining interpolation types, the derivative and
integral interfaces, and the caching options — must be imported from
DataInterpolations directly. `Curvefit` uses CurveFit.jl's nonlinear least-squares
algorithms, which DataDrivenDiffEq loads when it provides this reexport. DataDrivenDiffEq's
own kernel-based collocation
(`EpanechnikovKernel`, `GaussianKernel`, `collocate_data`, …) is documented under
[Utilities](utils.md).

## Data processing — owned by MLUtils.jl and StatsBase.jl

`DataProcessing` and `DataNormalization` (see [Solvers](solvers/common.md)) are thin
wrappers over upstream types, and the wrapped names are reexported so they can be
named at the call site:

  - `splitobs`, `DataLoader` — the train/test split and batching used by
    `DataProcessing`. Owned and documented by
    [MLUtils](https://juliaml.github.io/MLUtils.jl/stable/).
  - `ZScoreTransform`, `UnitRangeTransform` — the normalizations accepted by
    `DataNormalization`. Owned and documented by
    [StatsBase](https://juliastats.org/StatsBase.jl/stable/).

## Solving — owned by CommonSolve.jl

  - `solve` — the common solve entry point, owned by
    [CommonSolve](https://github.com/SciML/CommonSolve.jl) and shared across the SciML
    ecosystem.

## Sublibraries

[DataDrivenSR](libs/datadrivensr/symbolic_regression.md) reexports the
SymbolicRegression entry points needed to configure `EQSearch`; see that page.
DataDrivenDMD, DataDrivenLux and DataDrivenSparse reexport nothing.

## Keeping this page in sync

This list, the reexport `export` blocks at the bottom of `src/DataDrivenDiffEq.jl`,
and the `REEXPORTS` tuple in `test/qa/qa.jl` are the same list in three places.
`test/qa/qa.jl` checks that every approved name is actually reachable from
`using DataDrivenDiffEq`.
