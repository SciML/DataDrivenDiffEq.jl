# Basis

```@docs
Basis
```

## Generators

```@docs
monomial_basis
polynomial_basis
sin_basis
cos_basis
fourier_basis
chebyshev_basis
```
# [Koopman](@id koopman)

Since the results provided by [`DMD-like`](@ref koopman_algorithms) have special information, they have a separate subtype.

```@docs
Koopman
```

## Functions
```@docs
is_discrete
is_continuous
DataDrivenDiffEq.eigen
DataDrivenDiffEq.eigvals
DataDrivenDiffEq.eigvecs
modes
frequencies
operator
generator
updatable
is_stable
update!
```
