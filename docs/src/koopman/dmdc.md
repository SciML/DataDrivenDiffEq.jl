# Dynamic Mode Decomposition with control

[Dynamic Mode Decomposition with Control](https://epubs.siam.org/doi/abs/10.1137/15M1013857) is a method for
generating an approximating linear differential equation in a chosen basis of observables.
If `X` and `Y` are data matrices containing points of the same trajectory and `U` containing the exogenuos inputs
acting on that trajectory, `DMDc` approximates

```math
G = Y~\left[ \begin{array}{c} X \\ U \end{array} \right]^{\dagger} = \left[K ~B \right]
```

where ``\dagger`` denotes the Moore-Penrose pseudo inverse and `K` is the approximation of the Koopman operator and `B` the linear input map.

`DMDc` approximates *discrete time systems* with inputs ``y`` of the form

```math
u_{i+1} = K ~ u_{i} ~+ ~B ~ y_{i}
```

`gDMDc` approximates *continuous time systems* with inputs ``y`` of the form

```math
\frac{d}{dt}u =  K_{G} ~ u + B ~ y
```

where ``K_{G}`` is the generator of the Koopman operator.

## Functions

```@docs
DMDc
gDMDc
```
