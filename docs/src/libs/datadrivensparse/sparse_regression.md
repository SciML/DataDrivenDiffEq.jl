# DataDrivenSparse

DataDrivenSparse provides a universal framework to infer system of equations using sparse regression. Assume the system:

```math
y_{i} = f(x_{i}, p, t_i, u_{i})
```

Then might be able to express the unknown function $f$ as a linear combination of basis elements $\varphi_i : \mathbb R^{n_x} \times \mathbb R^{n_p} \times \mathbb R \times \mathbb R^{n_u} \mapsto \mathbb R$ .

```math
y_i = \sum_{j=1}^k \xi_k ~ \varphi_k\left(x_i, p, t_i, u_i \right)
```

And simply solve the least squares problem

```math
\Xi' = \min_{\Xi} \lVert Y - \Xi \varPhi \rVert_2^2
```

In the simplest case, we could use a Taylor expansion. However, if we want interpretable results, we need a key ingredient: sparsity! So, instead we aim to solve the problem

```math
\Xi' = \min_{\Xi} \lVert\Xi \rVert_0 \\
\text{s.t.} \qquad \Xi \varPhi =  Y
```

In its original version or via sufficient relaxation of the $L_0$ norm. 

Similarly, implicit problems of the form 

```math
f(y_i, x_i, p, t_i, u_i) = 0
```

can be solved using an [`ImplicitOptimizer`](@ref). Similar to the formulation above, we try to solve the corresponding optimization problem

```math
\Xi' = \min_{\Xi} \lVert\Xi \rVert_0 \\
\text{s.t.} \qquad \Xi \varPhi_y =  0
```

Where the matrix of evaluated basis elements $\varPhi_y \in \mathbb R^{\lvert \varphi \rvert} \times \mathbb R^{m}$ now may also contain basis functions which are dependent on the target variables $y \in \mathbb R^{n_y}$.

!!! warning "Tuning parameters for sparse regression"
    The algorithms used by `DataDrivenSparse` are sensible to the tuning of the hyperparameters! These are problem and coefficient specific, e.g. depend on the data and the unknown equations. While the examples used here are designed to work well, the used settings are not guaranteed to lead to success on other problems. User who want to explore the space of possible hyperparameters further might be interested in using [Hyperopt.jl](https://github.com/baggepinnen/Hyperopt.jl).

## [Algorithms](@id sparse_algorithms)

```@docs
STLSQ
ADMM
SR3
ImplicitOptimizer
```

## [Proximal Operators](@id proximal_operators)

```@docs
SoftThreshold
HardThreshold
ClippedAbsoluteDeviation
```
