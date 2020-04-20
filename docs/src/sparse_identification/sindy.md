## Sparse Identification of Nonlinear Dynamics

`SInDy` is the [method for generating sparse sets of equations](https://www.pnas.org/content/113/15/3932)
from a chosen basis. The function call is:

```julia
dudt = SInDy(data, dx, basis)
```

where `data` is a matrix of observed values (each column is a timepoint,
each row is an observable), `dx` is the matrix of derivatives of the observables,
`basis` is a `Basis`. This function will return a `Basis` constructed via a
sparse regression over initial `basis`.

`DataDrivenDiffEq` comes with some sparsifying regression algorithms (of the
abstract type `AbstractOptimiser`). Currently these are `STRRidge(threshold)`
from the [original paper](https://www.pnas.org/content/113/15/3932), a custom lasso
implementation via the alternating direction method of multipliers `ADMM(threshold, weight)`
and the `SR3(threshold, relaxation, proxoperator)` for [sparse relaxed regularized regression](https://arxiv.org/pdf/1807.05411.pdf). Here `proxoperator` can be any norm defined
via [ProximalOperators](https://github.com/kul-forbes/ProximalOperators.jl).
The `SInDy` algorithm can be called with all of the above via

```julia
opt = SR3()
dudt = SInDy(data, dx, basis, maxiter = 100, opt = opt)
```

In most cases, `STRRidge` works fine with little iterations (passed in via the `maxiter` argument ).
For larger datasets, `SR3` is in general faster even though it requires more iterations to converge.

Additionally, the boolean arguments `normalize` and `denoise` can be passed, which normalize the data matrix
or reduce it via the [optimal threshold for singular values](http://arxiv.org/abs/1305.5870).
