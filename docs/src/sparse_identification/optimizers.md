# Optimizer

`DataDrivenDiffEq` comes with some implementations for sparse regression included. All of these are stored inside the
`DataDrivenDiffEq.Optimize` package and extend the `AbstractOptimizer`, if an explicit optimization is needed or the `AbstractSubspaceOptimizer` for an implicit problem (where the solution is within the nullspace).

## Functions

```@docs
STLSQ
ADMM
SR3
DataDrivenDiffEq.Optimize.ADM
```


## Implementing New Optimizer

Similarly to [Algorithms for Estimation](@ref), the extension of optimizers is more or less straightforward.
Suppose you want to define a new optimizer `MyOpt`, which should solve ``A~X = Y`` for a sparse ``X``.

```julia
mutable struct MyOpt <: DataDrivenDiffEq.Optimize.AbstractOptimizer
    threshold
end
```

To use `MyOpt` within `SINDy`, an `init!` function has to be implemented.

```julia
function init!(X::AbstractArray, o::MyOpt, A::AbstractArray, Y::AbstractArray)
    X .=  A \ Y
end
```

To perform thresholding - and use maybe for searching the right threshold - a setter and getter is required:

```julia
set_threshold!(opt::MyOpt, threshold) = opt.threshold .= threshold

get_threshold(opt::MyOpt) = opt.threshold
```

And, at last, the method which fits the data and returns the iterations needed:

```julia
function fit!(X::AbstractArray, A::AbstractArray, Y::AbstractArray, opt::MyOpt; maxiter, convergence_error)
    # Compute awesome stuff here
    return iterations
end
```
