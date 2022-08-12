# [Solve](@id solve)

All algorithms have been combined under a single API to match the interface of other SciML packages. Thus, you can simply define a Problem, and then seamlessly switch between solvers. 

+ DataDrivenDMD for Koopman based inference
+ DataDrivenSparse for sparse regression based inference
+ DataDrivenSymbolicRegression for interfacing SymbolicRegression.jl

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
    on sufficiently large basis functions. By default eval_expression=false.

## Solving the Problem

After defining a [`problem`](@ref problem), we choose a method to [`solve`](@ref solve) it. Depending on the input arguments and the type of problem, the function will return a result derived the algorithm of choice. Different options can be provided, depending on the inference method, for options like rounding, normalization, or the progress bar. A [`Basis`](@ref) can be used for lifting the measurements.

```julia
# Use a Koopman based inference
res = solve(problem, DMDSVD(), kwargs...)
# Use a sparse identification
res = solve(problem, basis, STLQS(), kwargs...)
```

The [`DataDrivenSolution`](@ref) `res` contains a `result` which is the inferred system and a [`Basis`](@ref), `metrics` which is a `NamedTuple` containing different metrics of the inferred system. These can be accessed via:

```julia
# The inferred system
system = result(res)
# The metrics
m = metrics(res)
```

Since the inferred system is a parametrized equation, the corresponding parameters can be accessed and returned via

```julia
# Vector
ps = parameters(res)
# Parameter map
ps = parameter_map(res)
```


