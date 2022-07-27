"""
$(TYPEDEF)

The solution to a `DataDrivenProblem` derived via a certain algorithm.
The solution is represented via an `AbstractBasis`, which makes it callable.

# Fields
$(FIELDS)

# Note

The L₂ norm error, AIC and coefficient of determinantion get only computed, if eval_expression is set to true or
if the solution can be interpreted as a linear regression result.
"""
struct DataDrivenSolution{T} <: AbstractDataDrivenSolution
    "The basis representation of the solution"
    basis::AbstractBasis
    "Returncode"
    retcode::Symbol
    "Algorithm"
    alg::AbstractDataDrivenAlgorithm
    "Original output of the solution algorithm"
    out::AbstractDataDrivenResult
    "Problem"
    prob::AbstractDataDrivenProblem
    "Residual sum of squares"
    residuals::T
    "Degrees of freedom"
    dof::Int
end

_retcode(::ErrorDataDrivenResult) = :Failed

function DataDrivenSolution(b::Basis, p::AbstractDataDrivenProblem, 
    alg::AbstractDataDrivenAlgorithm = ZeroDataDrivenAlgorithm(), 
    result::AbstractDataDrivenResult = ErrorDataDrivenResult(), 
    )
    rss = sum(abs2, get_target(p) .- b(p))
    return DataDrivenSolution{eltype(p)}(
        b, 
        _retcode(result), 
        ZeroDataDrivenAlgorithm(),
        ErrorDataDrivenResult(),
        p,
        rss,
        length(parameters(b))
    )
end

(r::DataDrivenSolution)(args...) = r.basis(args...)

Base.show(io::IO, ::DataDrivenSolution{T}) where T = show(io, "DataDrivenSolution{$T}")

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
            println(io, "   $(v[1]) : $(v[2])")
        end
    end

    return
end

StatsBase.dof(sol::DataDrivenSolution) = getfield(sol, :dof)
StatsBase.rss(sol::DataDrivenSolution) = getfield(sol, :residuals)

StatsBase.loglikelihood(sol::DataDrivenSolution) = begin
    -nobs(sol)/2*log.(rss(sol)/nobs(sol))
end

StatsBase.nobs(sol::DataDrivenSolution) = begin
    prod(size(get_target(getfield(sol, :prob))))
end

@views StatsBase.nullloglikelihood(sol::DataDrivenSolution) = begin
    Y = get_target(getfield(sol, :prob))
    -nobs(sol)/2*log(sum(abs2, Y .- mean(Y, dims = 2))/nobs(sol))
end

StatsBase.r2(sol::DataDrivenSolution) = r2(sol, :CoxSnell)

@views StatsBase.summarystats(sol::DataDrivenSolution) = begin
    p = getfield(sol, :prob)
    map(summarystats, eachrow(get_target(p) .- sol(p)))
end


"""
$(SIGNATURES)

Returns the original [`DataDrivenProblem`](@ref).
"""
get_problem(r::DataDrivenSolution) = getfield(r, :prob)


"""
$(SIGNATURES)

Returns the recovered [`Basis`](@ref).
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
get_result(r::DataDrivenSolution) = getfield(r, :out)

"""
$(SIGNATURES)

Assert the result of the [`DataDrivenSolution`] and returns `true` if successful, `false` otherwise.
"""
is_converged(r::DataDrivenSolution) = getfield(r, :retcode) == :Success


