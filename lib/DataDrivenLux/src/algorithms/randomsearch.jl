@with_kw struct RandomSearch{F, A, L, P, O} <: AbstractDAGSRAlgorithm
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
    "Processes to use"
    procs::P = nothing
    "Random seed"
    rng::Random.AbstractRNG = Random.default_rng()
    "Optim optimiser"
    optimizer::O = LBFGS()
    "Optim options"
    optim_options::Optim.Options = Optim.Options()
end

Base.summary(io::IO, ::RandomSearch) = print(io, "RandomSearch()")

function update(ps::NamedTuple, model, alg::RandomSearch, candidates::AbstractVector,
                keeps::AbstractVector, X::AbstractArray, Y::AbstractArray)
    return ps
end

function init_cache(x::RandomSearch, X::AbstractArray{T}, Y::AbstractArray{T}) where {T}
    @unpack n_layers, functions, arities, skip, rng, populationsize = x
    @unpack optimizer, optim_options, loss = x
    # Derive the model
    model = LayeredDAG(size(X, 1), size(Y, 1), n_layers, arities, functions, skip = skip)
    ps, st = Lux.setup(rng, model)
    # Derive the candidates     
    candidates = map(1:populationsize) do i
        c = ConfigurationCache(model, ps, st, X, Y)
        c = optimize_configuration!(c, model, ps, X, Y, optimizer, optim_options)
    end

    keeps = zeros(Bool, populationsize)
    ages = zeros(Int, populationsize)
    sorting = sortperm(candidates, by = loss)

    return SearchCache{typeof(x), populationsize, typeof(model), typeof(ps)}(x, candidates,
                                                                             ages, sorting,
                                                                             keeps, 0,
                                                                             model, ps)
end
