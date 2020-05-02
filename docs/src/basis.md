## Basis

Many of the methods require the definition of a `Basis` on observables or
functional forms. A `Basis` is generated via:

```julia
Basis(h, u, parameters = [], iv = nothing)
```

where `h` is either a vector of ModelingToolkit `Operation`s for the valid functional
forms or a general function with the typical DiffEq signature `h(u,p,t)` which can be used with an  `Operation` or vector of `Operation`. `u` are the ModelingToolkit `Variable`s used to describe the Basis, and
`parameters` are the optional ModelingToolkit `Variable`s used to describe the
parameters in the basis elements. `iv` represents the independent variable of the system, the time.

```@docs
Basis
```

## Example

We start by crearting some `Variables` and `Parameters` using `ModelingToolkit`.
```@example basis
using LinearAlgebra
using DataDrivenDiffEq
using Plots
using ModelingToolkit

@variables u[1:3]
@parameters w[1:2]
```

To define a basis, simply write down the equations you want to be included as a
`Vector{Operation}`. Possible used parameters have to be given to the constructor.
```@example basis
h = [u[1]; u[2]; cos(w[1]*u[2]+w[2]*u[3])]
b = Basis(h, u, parameters = w)
```
What can a `Basis` do? Can it do stuff? Let's find out!

`Basis` are callable with the signature of functions to be used in `DifferentialEquations`.
So, the function value at a single point looks like:
```@example basis
x = b([1;2;3])
```
Or in place
```@example basis
dx = similar(x)
b(dx, [1;2;3])
```
Notice that since we did not use any numerical values for the parameters. The basis than uses the symbolic values in the result.

To use numerical values, simply pass this on in the function call. Here, we evaluate over a trajectory with two parameters and 40 timestamps.

```@example basis
X = randn(3, 40)
Y = b(X, [2;4], 0:39)
nothing # hide
```

Suppose we want to add another equation, say `sin(u[1])`. A `Basis` behaves like an array, so we can simply

```@example basis
push!(b, sin(u[1]))
size(b)
```
To ensure that a basis is well behaved, functions already present are not included again.

```@example basis
push!(b, sin(u[1]))
size(b)
```

We can also define functions of time and add them

```@example basis
t = independent_variable(b)
push!(b, cos(t*π))
println(b)
```

Additionally, we can iterate over a `Basis` using `[eq for eq in basis]` or index specific equations like `basis[2]`.

We can also chain `Basis` via just using it in the constructor

```@example basis
@variables x[1:2]
y = [sin(x[1]); cos(x[1]); x[2]]
t = independent_variable(b)
b2 = Basis(b(y, parameters(b), t), x, parameters = w, iv = t)
println(b2)
```

You can also use `merge` to create the union of two `Basis`:

```@example basis
b3 = merge(b, b2)
println(b3)
```

Which combines all the used variables and parameters ( and assume the same independent_variable ):

```@example basis
variables(b)
```

```@example basis
parameters(b)
```

If you have a function already defined as pure code, you can use this also
to create a `Basis`. Only the signature has to be consistent, so use `f(u,p,t)`.

```@example basis
f(u, p, t) = [u[1]; u[2]; cos(p[1]*u[2]+p[2]*u[3])]
b_f = Basis(f, u, parameters = w)
println(b_f)
```

This works for every function defined over `Operation`s. So to create a `Basis` from a `Flux` model, simply extend the activations used :

```julia
using Flux
NNlib.σ(x::Operation) = 1 / (1+exp(-x))

c = Chain(Dense(3,2,σ), Dense(2, 1, σ))
ps, re = Flux.destructure(c)

@parameters p[1:length(ps)]

g(u, p, t) = re(p)(u)
b = Basis(g, u, parameters = p)
```


## Functions

```@docs
parameters
variables
DataDrivenDiffEq.independent_variable
jacobian
dynamics
push!
deleteat!
merge
merge!
```
