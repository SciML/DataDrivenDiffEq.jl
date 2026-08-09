# [Solutions](@id datadrivensolution)

```@docs
DataDrivenSolution
DDReturnCode
```

## API

```@docs
get_problem
get_basis
get_algorithm
get_results
is_converged
```

## Statistical interface

```@docs
StatsAPI.dof(::DataDrivenSolution)
StatsAPI.rss(::DataDrivenSolution)
StatsAPI.loglikelihood(::DataDrivenSolution)
StatsAPI.nobs(::DataDrivenSolution)
StatsAPI.nullloglikelihood(::DataDrivenSolution)
StatsAPI.r2(::DataDrivenSolution)
StatsBase.summarystats(::DataDrivenSolution)
```
