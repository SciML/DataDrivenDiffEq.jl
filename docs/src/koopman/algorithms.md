# Algorithms for Estimation

There are different variants of estimation of the [Koopman Operator](@ref koopman_operator), see e.g., [here](http://www.aimsciences.org/journals/displayArticlesnew.jsp?paperID=10631), [here](https://link.springer.com/article/10.1007/s00332-015-9258-5) or [here](https://arxiv.org/abs/1611.06664).

Currently, `DataDrivenDiffEq` implements the following `AbstractKoopmanAlgorithms` to use with `DMD`, `EDMD`, and `DMDc`.


## Functions

```@docs
DMDPINV
DMDSVD
TOTALDMD
```

## Implementing New Algorithms

Is pretty straightforward. The implementation of `DMDPINV` looks like:

```julia

mutable struct DMDPINV <: AbstractKoopmanAlgorithm end;

(x::DMDPINV)(X::AbstractArray, Y::AbstractArray) = Y / X

```

So, right now, all you have to do is to implement a struct which is callable with the data matrices `X` and `Y`. Possible Parameters should be stored in the fields of the algorithm.
