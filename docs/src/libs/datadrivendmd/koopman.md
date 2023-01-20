# DataDrivenDMD

DataDrivenDMD provides operator-based inference. If we assume the following structure of a discrete dynamical system

```math
x_{i+1} = f(x_{i}, p, t, u_{i})
```

Then a valid Koopman representation states, that the system can be expressed as

```math
\varphi_{i+1} = \mathcal K \circ \varphi_i
```

where $\mathcal K$ denotes the Koopman operator, which is linear. However, this comes at the price of lifting the original state space $x \in \mathbb R^{n_x}$ into its observables $\varphi \in \mathbb C^{n_\varphi}$ with $n_x \leq n_{\varphi} \leq \infty$ . The important and most crucial fact here is the last inequality. While Koopman stated that any dynamical system can be expressed this way, it might well be that it can only be done in infinite dimensions.

Luckily, we can approximate the operator via Dynamic Mode Decomposition:

```math
\hat \varphi_{i+1} \approx K \hat \varphi_i
```

with $K \in \mathbb C^{n_d \times n_d}$ being a simple matrix, not necessary limited to the complex domain. 
A similar result holds for time continuous systems in the form of the Koopman generator:

```math
\partial_t \hat \varphi \approx K_G \hat \varphi
```

## [Algorithms](@id koopman_algorithms)

```@docs
DMDPINV
DMDSVD
TOTALDMD
FBDMD
```

