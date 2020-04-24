# Koopman Operators

## Shared Functions

Common function for all `AbstractKoopmanOperator`s.

```@docs
operator
generator
inputmap
outputmap
is_discete
is_continouos
updateable
isstable
eigen
eigvals
eigvecs
modes
frequencies
```

## Linear Koopman Operators

```@docs
LinearKoopman
update!
```

## Nonlinear Koopman Operatos

```@docs
NonlinearKoopman
reduce_basis
update!
```
