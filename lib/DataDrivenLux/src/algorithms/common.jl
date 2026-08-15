"""
    CommonAlgOptions(; kwargs...)

Shared configuration for [`AbstractDAGSRAlgorithm`](@ref) implementations.
Concrete algorithms normally expose these keywords through their own constructor.

# Fields

$(FIELDS)

# Keywords

- `populationsize::Int`: number of candidate graphs retained in the population.
- `functions`: candidate unary and binary functions.
- `arities`: arity corresponding to each entry in `functions`.
- `n_layers::Int`: number of learned graph layers.
- `skip::Bool`: whether each layer receives skip connections.
- `simplex::AbstractSimplex`: map used for categorical path weights.
- `loss`: function used to rank candidates.
- `keep::Union{Real,Int}`: retained fraction or number of candidates.
- `use_protected::Bool`: whether unsafe symbolic operations are replaced by safe
  versions.
- `distributed::Bool`: whether candidate optimization uses worker processes.
- `threaded::Bool`: whether candidate optimization uses Julia threads.
- `rng::AbstractRNG`: random-number generator for graph sampling.
- `optimizer`: Optim.jl optimizer for continuous candidate parameters.
- `optim_options`: optional Optim.jl options object.
- `optimiser`: optional Optimisers.jl update rule for search parameters.
- `observed`: optional fixed or fitted observation model.
- `alpha::Real`: exponential-update coefficient used by cross-entropy search.

# Returns

Return a configuration object consumed by [`AbstractDAGSRAlgorithm`](@ref)
implementations.

# Example

```julia
options = CommonAlgOptions(populationsize = 20, n_layers = 2)
options.populationsize == 20
```
"""
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
    optim_options
    optimiser <: Union{Nothing, Optimisers.AbstractRule}
    observed <: Union{ObservedModel, Nothing}
    alpha::Real
end

@public CommonAlgOptions

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
        optim_options = nothing,
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
