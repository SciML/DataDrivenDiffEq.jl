# Problem Definition And Solution

As can be seen from the [introduction examples](@id Quickstart), [DataDrivenDiffEq.jl](https://github.com/SciML/DataDrivenDiffEq.jl) tries to structurize the workflow in a similar fashion to other [SciML](https://sciml.ai/) packages by defining a [`DataDrivenProblem`](@ref), dispatching on the `solve` command to return a [`DataDrivenSolution`](@ref).

A problem in the sense of identification, estimation or inference is defined by the data describing it. This data contains at least measurements of the states `X`, which would be sufficient to describe a `DiscreteDataDrivenProblem` with unit time steps similar to the [first example on dynamic mode decomposition](@ref Linear-Systems-via-Dynamic-Mode-Decomposition). Of course we can extend this to include time points `t`, consecutive measurements `X̃` at the next time point or control signals `U` or a function describing those `u(x,p,t)`. Additionally, any parameters `p` known a priori can be included in the problem. In practice, this looks like

```julia
problem = DiscreteDataDrivenProblem(X)
problem = DiscreteDataDrivenProblem(X, t)
problem = DiscreteDataDrivenProblem(X, t, X̃)
problem = DiscreteDataDrivenProblem(X, t, X̃, U = U)
problem = DiscreteDataDrivenProblem(X, t, X̃, U = U, p = p)
problem = DiscreteDataDrivenProblem(X, t, X̃, U = (x,p,t)->u(x,p,t))
```

Similarly, a `ContinuousDataDrivenProblem` would need at least measurements and time-derivatives (`X` and `DX`) or measurements, time information and a way to derive the time derivatives(`X`, `t` and a [Collocation](@ref) method). Again, this can be extended by including a control input as measurements or a function and possible parameters.

```julia
problem = ContinuousDataDrivenProblem(X, t, InterpolationMethod())
problem = ContinuousDataDrivenProblem(X, DX = DX)
problem = ContinuousDataDrivenProblem(X, t, DX = DX)
problem = ContinuousDataDrivenProblem(X, t, DX = DX, U = U)
problem = ContinuousDataDrivenProblem(X, t, DX = DX, U = U, p = p)
problem = ContinuousDataDrivenProblem(X, t, DX = DX, U = (x,p,t)->u(x,p,t))
```

You can also directly use a `DESolution` as an input to your [`DataDrivenProblem`](@ref):

```julia
problem = DataDrivenProblem(sol; kwargs...)
```

which evaluates the function at the specific timepoints `t` using the parameters `p` of the original problem instead of
using the interpolation. If you want to use the interpolated data, add the additional keyword `use_interpolation = true`.

Next up, we choose a method to `solve` the [`DataDrivenProblem`](@ref). Depending on the input arguments and the type of problem, the function will return a result derived via [`Koopman`](@ref) or [`Sparse Optimization`](@ref) methods. Different options can be provided as well as a [`Basis`](@ref) used for lifting the measurements, to control different options like rounding, normalization or the progressbar depending on the inference method. Possible options are provided [below](@ref optional_arguments).

```julia
# Use a Koopman based inference
res = solve(problem, DMDSVD(), kwargs...)
# Use a sparse identification
res = solve(problem, basis, STLQS(), kwargs...)
```

The [`DataDrivenSolution`](@ref) `res` contains a `result` which is the infered system and a [`Basis`](@ref), `metrics` which is a `NamedTuple` containing different metrics like the L2 error of the infered system with the provided data and the [`AICC`](@ref). These can be accessed via

```julia
# The infered system
system = result(res)
# The metrics
m = metrics(res)
m.Sparsity # No. of active terms / nonzero coefficients
m.Error # L2 Error of all data
m.Errors # Individual error of the different data rows
m.AICC # AICC
m.AICCs # ....
```

Since the infered system is a parametrized equation, the corresponding parameters can be accessed and returned via

```julia
# Vector
ps = parameters(res)
# Parameter map
ps = parameter_map(res)
```

## [Optional Arguments](@id optional_arguments)

!!! info
The keyword argument `eval_expression` controls the function creation
behavior. `eval_expression=true` means that `eval` is used, so normal
world-age behavior applies (i.e. the functions cannot be called from
the function that generates them). If `eval_expression=false`,
then construction via GeneralizedGenerated.jl is utilized to allow for
same world-age evaluation. However, this can cause Julia to segfault
on sufficiently large basis functions. By default eval_expression=false.

Koopman based algorithms can be called without a [`Basis`](@ref), resulting in dynamic mode decomposition like methods, or with a basis for extened dynamic mode decomposition :

```julia
res = solve(problem, DMDSVD(), kwargs...)
res = solve(problem, basis, DMDSVD(), kwargs...)
```

If control signals are present, they get processed according to [this paper](https://epubs.siam.org/doi/abs/10.1137/15M1013857?mobileUi=0) for dynamic mode decomposition and [as described here](https://epubs.siam.org/doi/pdf/10.1137/16M1062296) for extended dynamic mode decomposition assuming a linear relationship on the operator.

Possible keyworded arguments include
+ `B` a linear mapping known a priori which maps the control signals onto the lifted states
+ `digits` controls the digits / rounding used for deriving the system equations (`digits = 1` would round `10.02` to `10.0`)
+ `operator_only` returns a `NamedTuple` containing the operator, input and output mapping and matrices used for updating the operator as described [here](https://arxiv.org/pdf/1406.7187.pdf)

!!! info
If `eval_expression` is set to `true`, the returning result of the Koopman based inference will not contain a parametrized equation, but rather use the numeric values of the operator/generator.

SINDy based algorithms can be called like :

```julia
res = solve(problem, basis, STLQS(), kwargs...)
res = solve(problem, basis, ImplicitOptimizer(), kwargs...)
```

Where control signals are included in the candidate basis.

For implicit optimizers, additionally an `Vector{Num}` of variables correspondng to ``u_t`` for implicitly defined
equations ``f(u_t, u, p, t) = 0`` can be passed in

```julia
res = solve(problem, basis, ImplicitOptimizer(), implicits, kwargs...)
```

This indicates that we do not have coupling between two implicitly defined variables if not explicitly given. To elaborate this complicated sentence, consider the following:

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

Would allow solutions of the form `x = y*x + z` or `y = y*x + z` to occur while suppressing `x = y + z`. This is due to the reason that `y*x` includes both `x` and `y`, so the function will get included in the evaluation.

Possible keyworded arguments include
+ `normalize` normalizes the data matrix ``\\Theta`` such that each row ( correspondng to candidate functions) has a 2-norm of `1.0`
+ `denoise` applies optimal shrinking to the matrix ``\\Theta`` to remove the influence of noise
+ `maxiter` Maximum iterations of the used optimizer
+ `round` rounds according to the currently used threshold of the optimizer

!!! warning
    For additional keywords, have a look at the examples in the [`Quickstart`](@ref) section or into the source code (for now).


## Types
```@docs
DataDrivenProblem
DataDrivenSolution
```

## Functions

```@docs
result
parameters
parameter_map
metrics
output
algorithm
inputs
```
