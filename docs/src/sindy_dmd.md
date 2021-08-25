# Unifying SINDy and DMD

The SINDy and DMD algorithms have been combined under a single interface, to match the interface of other SciML packages. Thus, you can simply define a Problem, and then seamlessly switch between solvers.

See the individual descriptions below for how to call the traditional SINDy and DMD solvers within the unified interface.

## DMD
For dynamic mode decomposition, use `DMDSVD()` without a basis:

```julia
res = solve(problem, DMDSVD(), kwargs...)
```

## Extended DMD
For extended dynamic mode decomposition, use  `DMDSVD()` with a basis:

```julia
res = solve(problem, basis, DMDSVD(), kwargs...)
```

## DMD Optional Arguments
If control signals are present, they get processed according to [this paper](https://epubs.siam.org/doi/abs/10.1137/15M1013857?mobileUi=0) for dynamic mode decomposition and [as described here](https://epubs.siam.org/doi/pdf/10.1137/16M1062296) for extended dynamic mode decomposition assuming a linear relationship on the operator.

Possible keyword arguments include:
+ `B` a linear mapping known a priori which maps the control signals onto the lifted states
+ `digits` controls the digits / rounding used for deriving the system equations (`digits = 1` would round `10.02` to `10.0`)
+ `operator_only` returns a `NamedTuple` containing the operator, input and output mapping and matrices used for updating the operator as described [here](https://arxiv.org/pdf/1406.7187.pdf)

!!! info
    If `eval_expression` is set to `true`, the returning result of the Koopman based inference will not contain a parametrized equation, but rather use the numeric values of the operator/generator.


## SINDy
For Sparse Identification of Nonlinear Dynamics, use `STLQS()`:

```julia
res = solve(problem, basis, STLQS(), kwargs...)
```

## Implicit SINDy
For Sparse Identification of Nonlinear Dynamics, use `ImplicitOptimizer()`:

```julia
res = solve(problem, basis, ImplicitOptimizer(), kwargs...)
```

Where control signals are included in the candidate basis.

For implicit optimizers, additionally an `Vector{Num}` of variables corresponding to ``u_t`` for implicitly defined
equations ``f(u_t, u, p, t) = 0`` can be passed in

```julia
res = solve(problem, basis, ImplicitOptimizer(), implicits, kwargs...)
```

This indicates that we do not have coupling between two implicitly defined variables unless coupling is explicitly included. To elaborate, consider the following:

```julia
basis = Basis([x, y, z], [x,y,z])
res = solve(problem, basis, ImplicitOptimizer())
```

Would allow solutions of the form `x = y + z`.

```julia
basis = Basis([x, y, z], [x,y,z])
res = solve(problem, basis, ImplicitOptimizer(), [x,y])
```

Would exclude solutions of the form `x = z` or `y = z` since we declared `x` and `y` as implicit variables assuming they don't interact. However, defining

```julia
basis = Basis([x, y, z, x*y], [x,y,z])
res = solve(problem, basis, ImplicitOptimizer(), [x,y])
```

Would allow solutions of the form `x = y*x + z` or `y = y*x + z` to occur while suppressing `x = y + z`. This is because `y*x` includes both `x` and `y`, so the function will get included in the evaluation.

## SINDy Optional Arguments

Possible keyword arguments include
+ `normalize` normalizes the data matrix ``\\Theta`` such that each row ( corresponding to candidate functions) has a 2-norm of `1.0`
+ `denoise` applies optimal shrinking to the matrix ``\\Theta`` to remove the influence of noise
+ `maxiter` Maximum iterations of the used optimizer
+ `round` rounds according to the currently used threshold of the optimizer
