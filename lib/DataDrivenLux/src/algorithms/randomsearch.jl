@concrete struct RandomSearch <: AbstractDAGSRAlgorithm
    options <: CommonAlgOptions
end

"""
$(SIGNATURES)

Performs a random search over the space of possible solutions to the symbolic regression
problem.
"""
function RandomSearch(; populationsize = 100, functions = (sin, exp, cos, log, +, -, /, *),
        arities = (1, 1, 1, 1, 2, 2, 2, 2), n_layers = 1, skip = true, loss = aicc,
        keep = 0.1, use_protected = true, distributed = false, threaded = false,
        rng = Random.default_rng(), optimizer = LBFGS(), optim_options = Optim.Options(),
        observed = nothing, alpha = 0.999f0)
    return RandomSearch(CommonAlgOptions(;
        populationsize, functions, arities, n_layers, skip, simplex = Softmax(), loss,
        keep, use_protected, distributed, threaded, rng, optimizer,
        optim_options, optimiser = nothing, observed, alpha))
end

Base.print(io::IO, ::RandomSearch) = print(io, "RandomSearch")
Base.summary(io::IO, x::RandomSearch) = print(io, x)

# Randomsearch does not do anything
update_parameters!(::SearchCache) = nothing
