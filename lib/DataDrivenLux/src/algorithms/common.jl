@concrete struct CommonAlgOptions
    populationsize::Int
    functions
    arities
    n_layers::Int
    skip::Bool
    simplex <: AbstractSimplex
    loss
    keep <: Union{Real, Int}
    use_protected::Bool
    distributed::Bool
    threaded::Bool
    rng <: AbstractRNG
    optimizer
    optim_options <: Optim.Options
    optimiser <: Union{Nothing, Optimisers.AbstractRule}
    observed <: Union{ObservedModel, Nothing}
    alpha::Real
end

function CommonAlgOptions(;
        populationsize::Int = 100,
        functions = (sin, exp, cos, log, +, -, /, *),
        arities = (1, 1, 1, 1, 2, 2, 2, 2),
        n_layers::Int = 1,
        skip::Bool = true,
        simplex::AbstractSimplex = Softmax(),
        loss = aicc,
        keep::Union{Real, Int} = 0.1,
        use_protected::Bool = true,
        distributed::Bool = false,
        threaded::Bool = false,
        rng::AbstractRNG = Random.default_rng(),
        optimizer = LBFGS(),
        optim_options::Optim.Options = Optim.Options(),
        optimiser::Union{Nothing, Optimisers.AbstractRule} = nothing,
        observed::Union{ObservedModel, Nothing} = nothing,
        alpha::Real = 0.999f0
    )
    return CommonAlgOptions(
        populationsize, functions, arities, n_layers, skip, simplex, loss, keep,
        use_protected, distributed, threaded, rng, optimizer, optim_options,
        optimiser, observed, alpha
    )
end
