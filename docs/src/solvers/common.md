# [Solve](@id solve)

All algorithms have been combined under a single API to match the interface of other SciML packages. Thus, you can simply define a Problem, and then seamlessly switch between solvers. 

+ [DataDrivenDMD](@ref) for Koopman based inference
+ [DataDrivenSparse](@ref) for sparse regression based inference
+ [DataDrivenSR](@ref) for interfacing SymbolicRegression.jl

All of the above methods return a [`DataDrivenSolution`](@ref) if not enforced otherwise.

## [Common Options](@id common_options)

Many of the algorithms implemented directly in `DataDrivenDiffEq` share common options. These can be passed into the `solve` call via keyworded arguments and get collected into the `CommonOptions` struct, which is given below. 

```@docs
DataDrivenCommonOptions
```

!!! info
    The keyword argument `eval_expression` controls the function creation
    behavior. `eval_expression=true` means that `eval` is used, so normal
    world-age behavior applies (i.e. the functions cannot be called from
    the function that generates them). If `eval_expression=false`,
    then construction via GeneralizedGenerated.jl is utilized to allow for
    same world-age evaluation. However, this can cause Julia to segfault
    on sufficiently large basis functions. By default `eval_expression=false`.

## Solving the Problem

After defining a [`problem`](@ref problem), we choose a method to [`solve`](@ref solve) it. Depending on the input arguments and the type of problem, the function will return a result derived the algorithm of choice. Different options can be provided, depending on the inference method, for options like rounding, normalization, or the progress bar. An optional [`Basis`](@ref) can be used for lifting the measurements.

```julia
solution = solve(DataDrivenProblem, [basis], solver; kwargs...)
```

If no [`Basis`](@ref) is supported, a unit basis is derived from the problem data containing the states and controls of the system.

Or more concrete examples:

```julia
# Use a Koopman based inference without a basis
res = solve(problem, DMDSVD(); options = DataDrivenCommonOptions(), kwargs...)
# Use a sparse identification
res = solve(problem, basis, STLSQ(); options = DataDrivenCommonOptions(),  kwargs...)
```
As we can see above, the use of a [`Basis`](@ref) is optional to invoke the estimation process. Internally, a linear [`Basis`](@ref) will be generated based on the [`DataDrivenProblem`](@ref problem) containing the states and control inputs.

The [`DataDrivenSolution`](@ref) `res` contains a `result` which is the inferred system and a [`Basis`](@ref).

## Model Selection

Most estimation and model inference algorithms require hyperparameters ,e.g., the sparsity controlling penalty, train-test splits. To account for this, the keyword `selector` can be passed to the [`DataDrivenCommonOptions`](@ref). This allows the user to control the selection criteria and returns the **minimum** selector. 

Common choices for `selector` are `rss`, `bic`, `aic`, `aicc`, and `r2`. Given that each subresult of the algorithm extends the `StatsBase` api, we can also use different schemes like:

```julia
options = DataDrivenCommonOptions(
    selector = (x)->rss(x) / nobs(x)
    )
```

Which results in the mean squared error of the system.
