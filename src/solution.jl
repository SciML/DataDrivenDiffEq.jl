"""
$(TYPEDEF)

The solution to a `DataDrivenProblem` derived via a certain algorithm.
The solution is represented via an `Basis`, which makes it callable.

# Fields

$(FIELDS)
"""
struct DataDrivenSolution{T} <: AbstractDataDrivenSolution
    "The basis representation of the solution"
    basis::AbstractBasis
    "Returncode"
    retcode::DDReturnCode
    "Algorithm"
    alg::AbstractDataDrivenAlgorithm
    "Original output of the solution algorithm"
    out::Vector{<:AbstractDataDrivenResult}
    "Problem"
    prob::AbstractDataDrivenProblem
    "Residual sum of squares"
    residuals::T
    "Degrees of freedom"
    dof::Int
    """Internal problem"""
    internal_problem::InternalDataDrivenProblem
end

function DataDrivenSolution(b::AbstractBasis, p::AbstractDataDrivenProblem,
        alg::AbstractDataDrivenAlgorithm,
        result::Vector{<:AbstractDataDrivenResult},
        internal_problem::InternalDataDrivenProblem,
        retcode = DDReturnCode(2))
    ps = get_parameter_values(b)
    prob = remake_problem(p, p = ps)

    # Calculate residual sum of squares, handling potential type instability
    # when basis has no equations (which can result in Any-typed matrices)
    Y = get_implicit_data(prob)
    Ŷ = b(prob)
    T = eltype(p)
    residuals = Y .- Ŷ
    # Handle case where residuals may have element type Any
    # (e.g., when basis has no equations due to all-zero coefficients)
    if eltype(residuals) === Any
        if isempty(residuals)
            rss = zero(T)
        else
            rss = sum(abs2, T.(residuals))
        end
    else
        rss = sum(abs2, residuals)
    end

    return DataDrivenSolution{eltype(p)}(b,
        retcode,
        alg,
        result,
        prob,
        rss,
        length(parameters(b)),
        internal_problem)
end

(r::DataDrivenSolution)(args...) = r.basis(args...)

Base.show(io::IO, ::DataDrivenSolution{T}) where {T} = show(io, "DataDrivenSolution{$T}")

function Base.print(io::IO, r::DataDrivenSolution)
    show(io, r)
    print(io, " with $(length(r.basis)) equations and $(r.dof) parameters.\n")
    print(io, "Returncode: $(r.retcode)\n")
    print(io, "Residual sum of squares: $(r.residuals)")
    return
end

function Base.print(io::IO, r::DataDrivenSolution, fullview::DataType)
    fullview != Val{true} && return print(io, r)

    print(io, r)

    if length(r.parameters) > 0
        x = parameter_map(r)
        println(io, "Parameters:")
        for v in x
            println(io, "  $(v[1]) : $(v[2])")
        end
    end

    return
end

"""
$(SIGNATURES)

Returns the degrees of freedom of the `DataDrivenSolution`.
"""
StatsBase.dof(sol::DataDrivenSolution) = getfield(sol, :dof)

"""
$(SIGNATURES)

Returns the residual sum of squares of the `DataDrivenSolution`.
"""
StatsBase.rss(sol::DataDrivenSolution) = getfield(sol, :residuals)

"""
$(SIGNATURES)

Returns the log-likelihood of the `DataDrivenSolution` assuming a normal distributed error.
"""
function StatsBase.loglikelihood(sol::DataDrivenSolution)
    begin
        -nobs(sol) / 2 * log.(rss(sol) / nobs(sol))
    end
end

"""
$(SIGNATURES)

Returns the number of observations of the `DataDrivenSolution`.
"""
function StatsBase.nobs(sol::DataDrivenSolution)
    begin
        prod(size(get_implicit_data(getfield(sol, :prob))))
    end
end

"""
$(SIGNATURES)

Return the null log-likelihood of the `DataDrivenSolution`. This corresponds to a model only fitted with an
intercept and a normal distributed error.
"""
@views function StatsBase.nullloglikelihood(sol::DataDrivenSolution)
    begin
        Y = get_implicit_data(getfield(sol, :prob))
        -nobs(sol) / 2 * log(sum(abs2, Y .- mean(Y, dims = 2)) / nobs(sol))
    end
end

"""
$(SIGNATURES)

Return the coefficient of determination of the `DataDrivenSolution`.

## Note

Only implements `CoxSnell` based on the [`loglikelihood`](@ref) and [`nullloglikelihood`](@ref).
"""
StatsBase.r2(sol::DataDrivenSolution) = r2(sol, :CoxSnell)

"""
$(SIGNATURES)

Returns the `summarystats` for each row of the error for the `DataDrivenSolution`.
"""
@views function StatsBase.summarystats(sol::DataDrivenSolution)
    p = getfield(sol, :prob)
    map(summarystats, eachrow(get_implicit_data(p) .- sol(p)))
end

"""
$(SIGNATURES)

Returns the original `DataDrivenProblem`.
"""
get_problem(r::DataDrivenSolution) = getfield(r, :prob)

"""
$(SIGNATURES)

Returns the recovered `Basis`.
"""
get_basis(r::DataDrivenSolution) = getfield(r, :basis)

"""
$(SIGNATURES)

Returns the algorithm used to derive the solution.
"""
get_algorithm(r::DataDrivenSolution) = getfield(r, :alg)

"""
$(SIGNATURES)

Returns the original output of the algorithm.
"""
get_results(r::DataDrivenSolution) = getfield(r, :out)

"""
$(SIGNATURES)

Assert the result of the `DataDrivenSolution` and returns `true` if successful, `false` otherwise.
"""
is_converged(r::DataDrivenSolution) = getfield(r, :retcode) == DDReturnCode(1)

## Conversions to DE / ODESystem / DAE or OptimizationSystem here
