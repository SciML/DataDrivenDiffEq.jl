# Dynamic Mode Decomposition

The (Exact) [Dynamic Mode Decomposition](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/dynamic-mode-decomposition-of-numerical-and-experimental-data/AA4C763B525515AD4521A6CC5E10DBD4) is a method for
generating an approximating linear differential equation directly from the observed data.
If `X` and `Y` are data matrices containing points of the same trajectory, than `DMD` approximates

```math
K = Y~X^{\dagger}
```

where ``\dagger`` denotes the Moore-Penrose pseudo inverse and `K` is the approximation of the Koopman operator.

`DMD` approximates *discrete time systems* of the form

```math
u_{i+1} = K ~ u_{i}
```

`gDMD` approximates *continuous time systems* of the form

```math
\frac{d}{dt}u =  K_{G} ~ u
```

where ``K_{G}`` is the generator of the Koopman operator.

```@docs
DMD
gDMD
```
