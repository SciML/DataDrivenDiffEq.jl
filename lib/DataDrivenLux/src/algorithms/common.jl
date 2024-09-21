@kwdef @concrete struct CommonAlgOptions
    populationsize::Int = 100
    functions = (sin, exp, cos, log, +, -, /, *)
    arities = (1, 1, 1, 1, 2, 2, 2, 2)
    n_layers::Int = 1
    skip::Bool = true
    simplex <: AbstractSimplex = Softmax()
    loss = aicc
    keep <: Union{Real, Int} = 0.1
    use_protected::Bool = true
    distributed::Bool = false
    threaded::Bool = false
    rng <: AbstractRNG = Random.default_rng()
    optimizer = LBFGS()
    optim_options <: Optim.Options = Optim.Options()
    optimiser <: Union{Nothing, Optimisers.AbstractRule} = nothing
    observed <: Union{ObservedModel, Nothing} = nothing
    alpha::Real = 0.999f0
end
