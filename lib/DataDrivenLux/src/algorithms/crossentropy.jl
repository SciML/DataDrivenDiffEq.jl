@concrete struct CrossEntropy <: AbstractDAGSRAlgorithm
    options <: CommonAlgOptions
end

"""
$(SIGNATURES)

Uses the crossentropy method for discrete optimization to search the space of possible
solutions.
"""
function CrossEntropy(; populationsize = 100, functions = (sin, exp, cos, log, +, -, /, *),
        arities = (1, 1, 1, 1, 2, 2, 2, 2), n_layers = 1, skip = true, loss = aicc,
        keep = 0.1, use_protected = true, distributed = false, threaded = false,
        rng = Random.default_rng(), optimizer = LBFGS(), optim_options = Optim.Options(),
        observed = nothing, alpha = 0.999f0)
    return CrossEntropy(CommonAlgOptions(;
        populationsize, functions, arities, n_layers, skip, simplex = DirectSimplex(), loss,
        keep, use_protected, distributed, threaded, rng, optimizer,
        optim_options, optimiser = nothing, observed, alpha))
end

Base.print(io::IO, ::CrossEntropy) = print(io, "CrossEntropy()")
Base.summary(io::IO, x::CrossEntropy) = print(io, x)

function init_model(x::CrossEntropy, basis::Basis, dataset::Dataset, intervals)
    (; n_layers, arities, functions, use_protected, skip) = x.options

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
        skip, input_functions = variable_mask, x.options.simplex)
end

function update_parameters!(cache::SearchCache{<:CrossEntropy})
    p̄ = mean(map(cache.candidates[cache.keeps]) do candidate
        return ComponentVector(get_configuration(candidate.model.model, cache.p, candidate.st))
    end)
    alpha = cache.alg.options.alpha
    @. cache.p = alpha * cache.p + (true - alpha) * p̄
    return
end
