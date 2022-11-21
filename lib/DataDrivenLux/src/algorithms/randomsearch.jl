@with_kw struct RandomSearch{F, A, L, O} <: AbstractDAGSRAlgorithm
    "The number of candidates to track"
    populationsize::Int = 100
    "The functions to include in the search"
    functions::F = (sin, exp, cos, log, +, -, /, *)
    "The arities of the functions"
    arities::A = (1, 1, 1, 1, 2, 2, 2, 2)
    "The number of layers"
    n_layers::Int = 2
    "Include skip layers"
    skip::Bool = true
    "Evaluation function"
    loss::L = aicc
    "The number of candidates to keep in each iteration"
    keep::Union{Real, Int} = 0.1
    "Use distributed optimization and resampling"
    distributed::Bool = false
    "Random seed"
    rng::Random.AbstractRNG = Random.default_rng()
    "Optim optimiser"
    optimizer::O = LBFGS()
    "Optim options"
    optim_options::Optim.Options = Optim.Options()
end

Base.summary(io::IO, ::RandomSearch) = print(io, "RandomSearch()")

function update(ps::NamedTuple, model, alg::RandomSearch, candidates::AbstractVector,
                keeps::AbstractVector, dataset::Dataset, basis::Basis)
    return ps
end

