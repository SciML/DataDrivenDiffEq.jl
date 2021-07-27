# Symbolic Regression

Using [sparse regression](@ref sparse_optimization) limits the discovery to a generalized linear model where it is assumed that the nonlinear [basis](@ref) can capture the underlying function properly. Another approach is to use a general expression tree, which commonly encodes the function to discover as a binary tree where the nodes represent unary or binary operators acting on their children. `DataDrivenDiffEq` includes the following symbolic regression algorithms.

## SymbolicRegression

!!! warning
    This feature requires the explicit loading of [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl) in addition to `DataDrivenDiffEq`. It will _only_ be useable if loaded like:
    ```julia
    using DataDrivenDiffEq
    using SymbolicRegression
    ```

`DataDrivenDiffEq` provides an interface to [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl) to `solve` a [DataDrivenProblem](@ref):

```@example symbolic_regression_api
using DataDrivenDiffEq
using LinearAlgebra
using Random
using SymbolicRegression


Random.seed!(1223)
# Generate a multivariate function for SymbolicRegression
X = rand(2,20)
f(x) = [sin(x[1]); exp(x[2])]
Y = hcat(map(f, eachcol(X))...)

# Define the options
opts = EQSearch([+, *, sin, exp], maxdepth = 1, progress = false, verbosity = 0)

# Define the problem
prob = DirectDataDrivenProblem(X, Y)

# Solve the problem
res = solve(prob, opts, numprocs = 0, multithreading = false)
sys = result(res)
println(sys) #hide
```

Where `solve` is used with [`EQSearch`](@ref), which wraps [`Options`](https://astroautomata.com/SymbolicRegression.jl/stable/api/#Options) provided by [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl). Additional keyworded arguments are `max_iter = 10`, which defines the number of iterations, `weights` which weight the measurements of the dependent variable (e.g. `X`, `DX` or `Y` depending on the [DataDrivenProblem](@ref)), `numprocs` which indicates the number of processes to use, `procs` for use with manually setup processes, `multithreading = false` for multithreading and `runtests = true` which performs initial testing on the environment to check for possible errors. It mimics the behaviour of [`EquationSearch`](https://astroautomata.com/SymbolicRegression.jl/stable/api/#EquationSearch).

### Related Types

```@docs
EQSearch
```

## OccamNet

!!! warning
    This feature requires the explicit loading of [Flux.jl](https://fluxml.ai/) in addition to `DataDrivenDiffEq`. It will _only_ be useable if loaded like:
    ```julia
    using DataDrivenDiffEq
    using Flux
    ```

As introduced in [Interpretable Neuroevolutionary Models for Learning Non-Differentiable Functions and Programs
](https://arxiv.org/abs/2007.10784), `OccamNet` is a special form of symbolic regression which uses a probabilistic approach to equation discovery by using a feedforward multilayer neural network. In contrast to normal architectures, each layer's weights reflect the probability of which inputs to use. Additionally a set of activation functions is used, instead of a single function. Similar to simulated annealing, a temperature is included to control the exploration of possible functions.

`DataDrivenDiffEq` offers two main interfaces to `OccamNet`: a `Flux` based API with `Flux.train!` and a `solve(...)` function.

Consider the following example, where we want to discover a vector valued function:

```@example occamnet_flux
using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using Flux
using Random

# Due to random values
Random.seed!(1223)

# Generate a multivariate dataset
X = rand(2,10)/c
f(x) = [sin(π*x[2]+x[1]); exp(x[2])]
Y = hcat(map(f, eachcol(X))...)
```

Next, we define our network:

```@example occamnet_flux
net = OccamNet(2, 2, 3, Function[sin, +, *, exp], skip = true, constants = Float64[π])
```

Where `2,2,3` refers to input and output dimension and the number of layers _without the output layer_. We also define that each layer uses the functions `sin, +, *, exp` as activations and uses a `π` as a constant, which get concatenated to the input data. Additionally, `skip` indicates the useage of skip connections, which allow the output of each layer to be passed onto the output layer directly.

To train the network over `100` epochs using `ADAM`, we type
```@example occamnet_flux
Flux.train!(net, X, Y, ADAM(1e-2), 100, routes = 100, nbest = 3)
```

Under the hood, we select possible routes, `routes`, through the network based on the probability reflected by the [`ProbabilityLayer`](@ref) forming the network. From these we take the `nbest` candidates to train the parameters of the network, meaning increase the probability of those routes.

Lets have a look at some possible equations after the initial training. We can use `rand` to sample a route through the network, compute the output probability with `probability` and transform it into analytical equations by simply using `ModelingToolkit`s variables as input. The call `net(x, route)` uses the route to compute just the element on this path.

```@example occamnet_flux
@variables x[1:2]

for i in 1:10
  route = rand(net)
  prob = probability(net, route)
  eq = simplify.(net(x, route))
  print(eq , " with probability ",  prob, "\n")
end
```
We see the networks proposals are not very certain. Hence, we will train for some more epochs and look at the output again.

```@example occamnet_flux
Flux.train!(net, X, Y, ADAM(1e-2), 900, routes = 100, nbest = 3)

for i in 1:10
  route = rand(net)
  prob = probability(net, route)
  eq = simplify.(net(x, route))
  print(eq , " with probability ",  prob, "\n")
end
```

The network is quite certain about the equation now, which is in fact our unknown mapping. To extract the solution with the highest probability, we set the temperature of the underlying distribution to a very low value. In the limit of `t ↦ 0` we approach a Dirac distribution, hence extracting the most likely terms.

```@example occamnet_flux
set_temp!(net, 0.01)
route = rand(net)
prob = probability(net, route)
eq = simplify.(net(x, route))
print(eq , " with probability ",  prob, "\n")
```

The same procedure is automated in the `solve` function. Using the same data, we wrap the algorithms information in the [`OccamSR`](@ref) struct and define a [`DataDrivenProblem`](@ref):

```@example occamnet_flux
# Define the problem
ddprob = DirectDataDrivenProblem(X, Y)
# Define the algorithm
sr_alg = OccamSR(functions = Function[sin, +, *, exp], skip = true, layers = 3, constants = [π])
# Solve the problem
res = solve(ddprob, sr_alg, ADAM(1e-2), max_iter = 1000, routes = 100, nbest = 3)
println(res) #hide
```

Within `solve` the network is generated using the information provided by the [DataDrivenProblem](@ref) in form of states, control and independent variables as well as the specified options, followed by training the network and extracting the equation with the highest probability by setting the temperature as above. After computing additional metrics, a [DataDrivenSolution](@ref) is returned where the equations are transformed  into a [`Basis`](@ref) useable with `ModelingToolkit`.

The metrics can be accessed via:

```@example occamnet_flux
metrics(res)
```

and the resulting [`Basis`](@ref) by:

```@example occamnet_flux
result(res)
println(result(res)) #hide
```

!!! info
    Right now, the resulting basis is not using parameters, but raw numerical values.

### Related Types

```@docs
OccamNet
OccamSR
ProbabilityLayer
```

### Related Functions


```@docs
set_temp!
probability
logprobability
probabilities
logprobabilities
```
