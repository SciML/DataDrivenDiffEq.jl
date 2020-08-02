# Extended Dynamic Mode Decomposition

[Extended Dynamic Mode Decomposition](https://link.springer.com/article/10.1007/s00332-015-9258-5) is a method for
generating an approximating linear differential equation in a chosen basis of observables.
If `X` and `Y` are data matrices containing points of the same trajectory and `Ψ` is a basis, then `EDMD` approximates

```math
K = Ψ(Y)~Ψ(X)^{\dagger}
```

where ``\dagger`` denotes the Moore-Penrose pseudoinverse and `K` is the approximation of the [Koopman Operator](@ref koopman_operator).

`EDMD` approximates *discrete time systems* of the form

```math
\Psi(u_{i+1}) = K ~ \Psi(u_{i}) \\
u_{i+1} = C~\Psi(u_{i})
```

`gEDMD` approximates *continuous time systems* of the form

```math
\frac{d}{dt}\Psi(u) =  K_{G} ~ \Psi(u) \\
\frac{d}{dt}u = C~\frac{d}{dt}\Psi(u)
```

where ``K_{G}`` is the generator of the [Koopman Operator](@ref koopman_operator).
`C` is a matrix mapping from the observations back onto the states or differential states.

## Functions
```@docs
EDMD
gEDMD
```
