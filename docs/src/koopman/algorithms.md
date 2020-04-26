# Algorithms for Estimation

There are different different variants of estimation of the Koopman operator, see e.g. [here](), [here]() or []().

Currently, `DataDrivenDiffEq` implements the following `AbstractKoopmanAlgorithms` to use with `DMD`, `EDMD` and `DMDc`.


## Functions

```@docs
DMDPINV
DMDSVD
```

## Implementing New Algorithms

Is pretty straightforward. The implementation of `DMDMPINV` looks like

```julia

mutable struct DMDPINV <: AbstractKoopmanAlgorithm end;

(x::DMDPINV)(X::AbstractArray, Y::AbstractArray) = Y / X

```

So right now, all you have to do is to implement a struct which is callable with the data matrices `X` and `Y`. Possible Parameters should be stored in the fields of the algorithm.
