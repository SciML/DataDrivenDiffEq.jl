"""
$(TYPEDEF)

The solution to a `DataDrivenEnsembleProblem` derived via a certain algorithm.
The solution is represented via an `AbstractBasis`, which makes it callable.

# Fields
$(FIELDS)
"""
struct DataDrivenEnsembleSolution{L, A, O} <: AbstractDataDrivenSolution 
    "The basis representation of the solution"
    basis::AbstractBasis
    "Parameters of the solution"
    parameters::AbstractVecOrMat
    "Returncode"
    retcode::Symbol
    "Algorithm"
    alg::A
    "Weights"
    weights::AbstractWeights
    "Problem"
    prob::DataDrivenEnsemble
    "Individual results"
    results::AbstractVector{DataDrivenSolution{L,A,O}}
end

(r::DataDrivenEnsembleSolution)(args...) = r.basis(args...)

Base.summary(io::IO, s::DataDrivenEnsembleSolution) = begin
    print(io, "EnsembleSolution with $(length(s.results)) solutions.")
end

Base.show(io::IO, s::DataDrivenEnsembleSolution) = summary(io, s)

function Base.print(io::IO, r::DataDrivenEnsembleSolution)
    show(io, r)
    print(io, " with $(length(r.basis)) equations and $(length(r.parameters)) parameters.\n")
    print(io, "Returncode: $(r.retcode)\n")
    return
end

function Base.print(io::IO, r::DataDrivenEnsembleSolution, fullview::DataType)
    print(io, r)
    fullview != Val{true} && return 
    for res in r.results
        print(io, res, fullview)
    end
    return
end

"""
$(SIGNATURES)

Return the solution(s) of the `EnsembleSolution`.
"""
get_solution(r::DataDrivenEnsembleSolution) = getfield(r, :results)

get_solution(r::DataDrivenEnsembleSolution, id) = begin
    getindex(get_solution(r), id)
end

"""
$(SIGNATURES)

Return the weight(s) of the `EnsembleSolution`.
"""
get_weights(r::DataDrivenEnsembleSolution) = getfield(r, :weights)

get_weights(r::DataDrivenEnsembleSolution, id) = begin
    getindex(get_weights(r), id)
end

function metrics(r::DataDrivenEnsembleSolution)
    map(metrics, get_solution(r))
end

function metrics(r::DataDrivenEnsembleSolution, id)
    map(metrics, get_solution(r, id))
end

function DataDrivenEnsembleSolution(prob, results, success, b::Basis, alg::AbstractOptimizer; digits::Int = 10, by = :min, eval_expression = false, kwargs...)
    # Compute the averages
    Ξ, errors, λ = select_by(by, map(output, results[success]))
    
    # Assert continuity
    lhs, dt = assert_lhs(prob)

    sol , ps = construct_basis(round.(Ξ, digits = digits), b, implicit_variables(b), 
        lhs = lhs, dt = dt,
        is_implicit = isa(alg, AbstractSubspaceOptimizer) ,eval_expression = eval_expression
        )

    ps = isempty(parameters(b)) ? ps : vcat(prob.p, ps)

    return DataDrivenEnsembleSolution(
        sol, ps, :solved, alg, Weights([s ? 1.0 : 0.0 for s in success]), prob, [r for r in results]
    )
end


function DataDrivenEnsembleSolution(prob, results, success, b::Basis, alg::AbstractKoopmanAlgorithm; 
    operator_only = false, digits::Int = 10, by = :min, eval_expression = false, kwargs...)
    # Compute the averages
    Ξ, errors = select_by(by, map(output, results[success]))
    
    @unpack inds = k
    K, B, C, P, Q = k_
    operator_only && return (K = K, B = B, C = C, P = P, Q = Q)
    # Build parameterized equations, inds indicate the location of basis elements containing an input
    Ξ = zeros(eltype(C), size(C,2), length(b))


    Ξ[:, inds] .= real.(Matrix(K))
    if !isempty(B)
        Ξ[:, .! inds] .= B
    end

    # Assert continuity
    lhs, dt = assert_lhs(prob)
    
    bs, ps = construct_basis(round.(C*Ξ, digits = digits)', b, 
        lhs = lhs, dt = dt,
        eval_expression = eval_expression)


    res_ = Koopman(equations(bs), states(bs),
        parameters = parameters(bs),
        controls = controls(bs), iv = get_iv(bs),
        K = K, C = C, Q = Q, P = P, lift = get_f(b),
        is_discrete = is_discrete(prob),
        eval_expression = eval_expression)

    ps = isempty(parameters(b)) ? ps : vcat(prob.p, ps)

    
    return DataDrivenEnsembleSolution(
        sol, ps, :solved, alg, Weights([s ? 1.0 : 0.0 for s in success]), prob, results, eval_expression = eval_expression
    )
end