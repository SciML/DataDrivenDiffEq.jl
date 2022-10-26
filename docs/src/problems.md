# [Problems](@id problem)

```@docs
DataDrivenProblem
```
## Defining a Problem

Problems of identification, estimation, or inference are defined by data. These data contain at least measurements of the states `X`, which would be sufficient to describe a `[DiscreteDataDrivenProblem`](@ref) with unit time steps similar to the [first example on dynamic mode decomposition](@ref linear_discrete). Of course, we can extend this to include time points `t`, control signals `U` or a function describing those `u(x,p,t)`. Additionally, any parameters `p` known a priori can be included in the problem. In practice, this looks like:

```julia
problem = DiscreteDataDrivenProblem(X)
problem = DiscreteDataDrivenProblem(X, t)
problem = DiscreteDataDrivenProblem(X, t, U)
problem = DiscreteDataDrivenProblem(X, t, U, p = p)
problem = DiscreteDataDrivenProblem(X, t, (x,p,t)->u(x,p,t))
```

Similarly, a [`ContinuousDataDrivenProblem`](@ref) would need at least measurements and time-derivatives (`X` and `DX`) or measurements, time information and a way to derive the time derivatives(`X`, `t` and a [Collocation](@ref collocation) method). Again, this can be extended by including a control input as measurements or a function and possible parameters:

```julia
# Using available data
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

An additional type of problem is the [`DirectDataDrivenProblem`](@ref), which does not assume any kind of causal relationship. It is defined by `X` and an observed output `Y` in addition to the usual arguments:

```julia
problem = DirectDataDrivenProblem(X, Y)
problem = DirectDataDrivenProblem(X, t, Y)
problem = DirectDataDrivenProblem(X, t, Y, U)
problem = DirectDataDrivenProblem(X, t, Y, p = p)
problem = DirectDataDrivenProblem(X, t, Y, (x,p,t)->u(x,p,t), p = p)
```
## Concrete Types

```@docs
DiscreteDataDrivenProblem
ContinuousDataDrivenProblem
DirectDataDrivenProblem
```

# [Datasets](@id dataset)

```@docs
DataDrivenDataset
```

A `DataDrivenDataset` collects several [`DataDrivenProblem`s](@ref problem) of the same type but treads them as union used for system identification. 
## Concrete Types
```@docs
DiscreteDataset
ContinuousDataset
DirectDataset
```
