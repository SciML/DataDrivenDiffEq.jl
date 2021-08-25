# Getting Started

The workflow for [DataDrivenDiffEq.jl](https://github.com/SciML/DataDrivenDiffEq.jl) is similar to other [SciML](https://sciml.ai/) packages. You start by defining a [`DataDrivenProblem`](@ref) and then dispatch on the `solve` command to return a [`DataDrivenSolution`](@ref).

Outline of required elements and choices:
1. Define a problem using your Data
  + Data can be discrete, continuous, or direct.
2. Choose a basis
  + This is optional depending on which solver you choose.
3. Solve the problem
  + Many solvers exist, see the [docs](@ref koopman_algorithms).

## Basic usage

```julia
using DataDrivenDiffEq, ModelingToolkit

# The function we are trying to find
f(u) = u^2 + 4u + 4
X = f.(1:100) # Generate data
X = reshape(X, length(X), 1) # Reshape into a matrix

# Create a problem from the data
problem = DiscreteDataDrivenProblem(X)

# Choose a basis
@variables u[1:1]
using Symbolics: scalarize
u = scalarize(u)
basis = Basis(monomial_basis(u, 2), u)

# Solve the problem, using the solver of your choosing
res = solve(problem, basis, STLSQ())
```

## Defining a Problem

Problems of identification, estimation or inference are defined by the data. This data contains at least measurements of the states `X`, which would be sufficient to describe a `DiscreteDataDrivenProblem` with unit time steps similar to the [first example on dynamic mode decomposition](@ref Linear-Systems-via-Dynamic-Mode-Decomposition). Of course we can extend this to include time points `t`, control signals `U` or a function describing those `u(x,p,t)`. Additionally, any parameters `p` known a priori can be included in the problem. In practice, this looks like:

```julia
problem = DiscreteDataDrivenProblem(X)
problem = DiscreteDataDrivenProblem(X, t)
problem = DiscreteDataDrivenProblem(X, t, U)
problem = DiscreteDataDrivenProblem(X, t, U, p = p)
problem = DiscreteDataDrivenProblem(X, t, (x,p,t)->u(x,p,t))
```

Similarly, a `ContinuousDataDrivenProblem` would need at least measurements and time-derivatives (`X` and `DX`) or measurements, time information and a way to derive the time derivatives(`X`, `t` and a [Collocation](@ref) method). Again, this can be extended by including a control input as measurements or a function and possible parameters:

```julia
problem = ContinuousDataDrivenProblem(X, DX)
problem = ContinuousDataDrivenProblem(X, t, DX)
problem = ContinuousDataDrivenProblem(X, t, DX, U, p = p)
problem = ContinuousDataDrivenProblem(X, t, DX, (x,p,t)->u(x,p,t))
# Using collocation
problem = ContinuousDataDrivenProblem(X, t, InterpolationMethod())
problem = ContinuousDataDrivenProblem(X, t, GaussianKernel())
problem = ContinuousDataDrivenProblem(X, t, U, InterpolationMethod())
problem = ContinuousDataDrivenProblem(X, t, U, GaussianKernel(), p = p)
```

You can also directly use a `DESolution` as an input to your [`DataDrivenProblem`](@ref):

```julia
problem = DataDrivenProblem(sol; kwargs...)
```

which evaluates the function at the specific timepoints `t` using the parameters `p` of the original problem instead of
using the interpolation. If you want to use the interpolated data, add the additional keyword `use_interpolation = true`.

An additional type of problem is the `DirectDataDrivenProblem`, which does not assume any kind of causal relationship. It is defined by `X` and an observed output `Y` in addition to the usual arguments:

```julia
problem = DirectDataDrivenProblem(X, Y)
problem = DirectDataDrivenProblem(X, t, Y)
problem = DirectDataDrivenProblem(X, t, Y, U)
problem = DirectDataDrivenProblem(X, t, Y, p = p)
problem = DirectDataDrivenProblem(X, t, Y, (x,p,t)->u(x,p,t), p = p)
```

## Choosing a Basis

A basis is optional, depending on the solver and solution method you are using. For instance, for DMD, a basis is not required, but for SINDy using STLQS(), it is required.

A basis can be defined like:
```julia
@variables u[1:2]
Ψ = Basis([u; u[1]^2], u)
```

See the [tutorials](@ref real_world) for more complex examples of defining a Basis.

## Solving the Problem

Next up, we choose a method to `solve` the [`DataDrivenProblem`](@ref). Depending on the input arguments and the type of problem, the function will return a result derived via [`Koopman`](@ref) or [`Sparse Optimization`](@ref sparse_optimization) methods. Different options can be provided as well as a [`Basis`](@ref) used for lifting the measurements, to control different options like rounding, normalization or the progressbar depending on the inference method.

```julia
# Use a Koopman based inference
res = solve(problem, DMDSVD(), kwargs...)
# Use a sparse identification
res = solve(problem, basis, STLQS(), kwargs...)
```

The [`DataDrivenSolution`](@ref) `res` contains a `result` which is the inferred system and a [`Basis`](@ref), `metrics` which is a `NamedTuple` containing different metrics like the L2 error of the inferred system with the provided data and the [`AICC`](@ref). These can be accessed via:

```julia
# The inferred system
system = result(res)
# The metrics
m = metrics(res)
m.Sparsity # No. of active terms / nonzero coefficients
m.Error # L2 Error of all data
m.Errors # Individual error of the different data rows
m.AICC # AICC
m.AICCs # ....
```

Since the inferred system is a parametrized equation, the corresponding parameters can be accessed and returned via

```julia
# Vector
ps = parameters(res)
# Parameter map
ps = parameter_map(res)
```

!!! info
    The keyword argument `eval_expression` controls the function creation
    behavior. `eval_expression=true` means that `eval` is used, so normal
    world-age behavior applies (i.e. the functions cannot be called from
    the function that generates them). If `eval_expression=false`,
    then construction via GeneralizedGenerated.jl is utilized to allow for
    same world-age evaluation. However, this can cause Julia to segfault
    on sufficiently large basis functions. By default eval_expression=false.
