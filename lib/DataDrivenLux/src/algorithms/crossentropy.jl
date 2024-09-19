"""
$(TYPEDEF)

Uses the crossentropy method for discrete optimization to search the space of possible solutions.

# Fields
$(FIELDS)
"""
@kwdef struct CrossEntropy{F, A, L, O} <: AbstractDAGSRAlgorithm
    "The number of candidates to track"
    populationsize::Int = 100
    "The functions to include in the search"
    functions::F = (sin, exp, cos, log, +, -, /, *)
    "The arities of the functions"
    arities::A = (1, 1, 1, 1, 2, 2, 2, 2)
    "The number of layers"
    n_layers::Int = 1
    "Include skip layers"
    skip::Bool = true
    "Evaluation function to sort the samples"
    loss::L = aicc
    "The number of candidates to keep in each iteration"
    keep::Union{Real, Int} = 0.1
    "Use protected operators"
    use_protected::Bool = true
    "Use distributed optimization and resampling"
    distributed::Bool = false
    "Use threaded optimization and resampling - not implemented right now."
    threaded::Bool = false
    "Random seed"
    rng::AbstractRNG = Random.default_rng()
    "Optim optimiser"
    optimizer::O = LBFGS()
    "Optim options"
    optim_options::Optim.Options = Optim.Options()
    "Observed model - if `nothing`is used, a normal distributed additive error with fixed variance is assumed."
    observed::Union{ObservedModel, Nothing} = nothing
    "Field for possible optimiser - no use for CrossEntropy"
    optimiser::Nothing = nothing
    "Update parameter for smoothness"
    alpha::Real = 0.999f0
end

Base.print(io::IO, ::CrossEntropy) = print(io, "CrossEntropy")
Base.summary(io::IO, x::CrossEntropy) = print(io, x)

function init_model(x::CrossEntropy, basis::Basis, dataset::Dataset, intervals)
    (; n_layers, arities, functions, use_protected, skip) = x

    # We enforce the direct simplex here!
    simplex = DirectSimplex()

    # Get the parameter mapping
    variable_mask = map(enumerate(equations(basis))) do (i, eq)
        return any(ModelingToolkit.isvariable, ModelingToolkit.get_variables(eq.rhs)) &&
               IntervalArithmetic.iscommon(intervals[i])
    end

    variable_mask = Any[variable_mask...]

    if use_protected
        functions = map(convert_to_safe, functions)
    end

    return LayeredDAG(length(basis), size(dataset.y, 1), n_layers, arities, functions;
        skip = skip, input_functions = variable_mask, simplex = simplex)
end

function update_parameters!(cache::SearchCache{<:CrossEntropy})
    (; candidates, keeps, p, alg) = cache
    (; alpha) = alg
    p̄ = mean(map(candidates[keeps]) do candidate
        return ComponentVector(get_configuration(candidate.model.model, p, candidate.st))
    end)
    cache.p .= alpha * p + (one(alpha) - alpha) .* p̄
    return
end
