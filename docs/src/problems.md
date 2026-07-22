# [Problems](@id problem)

```@docs
DataDrivenProblem
```

## Defining a Problem

Problems of identification, estimation, or inference are defined by data. These data contain at least measurements of the states `X`, which would be sufficient to describe a [`DiscreteDataDrivenProblem`](@ref) with unit time steps similar to the first example on dynamic mode decomposition. Of course, we can extend this to include time points `t`, control signals `U` or a function describing those `u(x,p,t)`. Additionally, any parameters `p` known a priori can be included in the problem. In practice, this looks like:

```julia
problem = DiscreteDataDrivenProblem(X)
problem = DiscreteDataDrivenProblem(X, t)
problem = DiscreteDataDrivenProblem(X, t, U)
problem = DiscreteDataDrivenProblem(X, t, U, p = p)
problem = DiscreteDataDrivenProblem(X, t, (x, p, t) -> u(x, p, t))
```

Similarly, a [`ContinuousDataDrivenProblem`](@ref) would need at least measurements and time-derivatives (`X` and `DX`) or measurements, time information, and a way to derive the time derivatives (`X`, `t`, and a [Collocation](@ref collocation) method). Again, this can be extended by including a control input as measurements or a function and possible parameters:

```julia
# Using available data
problem = ContinuousDataDrivenProblem(X, DX)
problem = ContinuousDataDrivenProblem(X, t, DX)
problem = ContinuousDataDrivenProblem(X, t, DX, U, p = p)
problem = ContinuousDataDrivenProblem(X, t, DX, (x, p, t) -> u(x, p, t))

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
problem = DirectDataDrivenProblem(X, t, Y, (x, p, t) -> u(x, p, t), p = p)
```

## Working with Real Data

When working with experimental data from files (e.g., CSV), the data must be formatted correctly before creating a problem. The key points are:

- **States `X`**: A matrix of shape `(n_states, n_timepoints)` where each column is a measurement at a time point
- **Times `t`**: A vector of length `n_timepoints`
- **Controls `U`**: Either a matrix of shape `(n_controls, n_timepoints)` or a function `(x, p, t) -> u_vector`

### Loading Data from CSV

```julia
using CSV, DataFrames

# Load your experimental data
df = CSV.read("experiment.csv", DataFrame)

# Extract time points
t = Vector(df.time)

# Extract state measurements (transpose so columns are time points)
X = permutedims(Matrix(df[:, [:x1, :x2]]))

# Extract control measurements
U = permutedims(Matrix(df[:, [:u1]]))

# Create the problem
prob = ContinuousDataDrivenProblem(X, t, U = U)
```

### Time-Varying Controls from Data

Control inputs can be specified in two ways:

1. **As measured data** (matrix): Use this when you have control values recorded at each time point
   ```julia
   U = [u1_at_t1 u1_at_t2 ... u1_at_tn;
        u2_at_t1 u2_at_t2 ... u2_at_tn]  # Shape: (n_controls, n_timepoints)
   prob = ContinuousDataDrivenProblem(X, t, U = U)
   ```

2. **As a function**: Use this when controls can be computed analytically or when you want to interpolate measured data
   ```julia
   # Using DataInterpolations.jl to create a continuous function from discrete data
   using DataInterpolations
   u_interp = LinearInterpolation(vec(U), t)
   control_func(x, p, t) = [u_interp(t)]
   prob = ContinuousDataDrivenProblem(X, t, U = control_func)
   ```

For a complete example, see [Using Real Data with Time-Varying Controls](@ref real_data_controls).

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

A `DataDrivenDataset` collects several [`DataDrivenProblem`](@ref problem)s of the same type but treats them as a union for system identification.

## Concrete Types

```@docs
DiscreteDataset
ContinuousDataset
DirectDataset
```

## API

These methods are defined for [`DataDrivenProblem`](@ref problem)s, but might be useful for developers.

```@docs
is_direct
is_discrete
is_continuous
has_timepoints
is_autonomous
is_parametrized
get_name
is_valid
@is_applicable
```
