"""
$(TYPEDEF)

Performs a random search over the space of possible solutions to the 
symbolic regression problem.

# Fields
$(FIELDS)
"""
@kwdef struct RandomSearch{F, A, L, O} <: AbstractDAGSRAlgorithm
    # "The number of candidates to track"
    # populationsize::Int = 100
    # "The functions to include in the search"
    # functions::F = (sin, exp, cos, log, +, -, /, *)
    # "The arities of the functions"
    # arities::A = (1, 1, 1, 1, 2, 2, 2, 2)
    # "The number of layers"
    # n_layers::Int = 1
    # "Include skip layers"
    # skip::Bool = true
    # "Simplex mapping"
    # simplex::AbstractSimplex = Softmax()
    # "Evaluation function to sort the samples"
    # loss::L = aicc
    # "The number of candidates to keep in each iteration"
    # keep::Union{Real, Int} = 0.1
    # "Use protected operators"
    # use_protected::Bool = true
    # "Use distributed optimization and resampling"
    # distributed::Bool = false
    # "Use threaded optimization and resampling - not implemented right now."
    # threaded::Bool = false
    # "Random seed"
    # rng::AbstractRNG = Random.default_rng()
    # "Optim optimiser"
    # optimizer::O = LBFGS()
    # "Optim options"
    # optim_options::Optim.Options = Optim.Options()
    # "Observed model - if `nothing`is used, a normal distributed additive error with fixed variance is assumed."
    # observed::Union{ObservedModel, Nothing} = nothing
    # "Field for possible optimiser - no use for Randomsearch"
    # optimiser::Nothing = nothing
end

Base.print(io::IO, ::RandomSearch) = print(io, "RandomSearch")
Base.summary(io::IO, x::RandomSearch) = print(io, x)

# Randomsearch does not do anything
function update_parameters!(::SearchCache)
    return
end
