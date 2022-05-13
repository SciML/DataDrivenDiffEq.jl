"""
$(TYPEDEF)

A collection of DataDrivenProblems used to concatenate different trajectories or experiments of the same underlying 
model with different realizations. This means, there is a model representing the superset of all possible parametrizations of the
underlying function. Hence, the parameters of each realization do not have to be equal.

Using `solve(::DataDrivenEnsemble, args...; kwargs...)` will result in an individual `solve` call for all problems contained. 
Right now, this is done sequentially.

Can be called with either a `NTuple` of problems or a `NamedTuple` of `NamedTuples`. 
Similar to the `DataDrivenProblem`, it has three constructors available:

+ `DirectDataEnsemble` for direct problems
+ `DiscreteDataEnsemble` for discrete problems
+ `ContinuousDataEnsemble` for continuous problems

# Fields
$(FIELDS)

# Signatures
$(SIGNATURES)
"""
struct DataDrivenEnsemble{N, U, C} <: AbstractDataDrivenProblem{N,U,C}
    """Name of the dataset"""
    name::Symbol
    """The problems"""
    probs::NTuple{M, AbstractDataDrivenProblem{N,U,C}} where M
end

# Constructor

function DataDrivenEnsemble(probs::Vararg{T, N}; name = gensym(:DDEnsemble), kwargs...) where {T <: AbstractDataDrivenProblem, N}
    return DataDrivenEnsemble(name, probs)
end

function DataDrivenEnsemble(solutions::Vararg{T, N}; name = gensym(:DDEnsemble), kwargs...) where {T <: DiffEqBase.DESolution, N}
    probs = map(solutions) do s
        DataDrivenProblem(s; kwargs...)
    end
    return DataDrivenEnsemble(probs..., name = name)
end

"""
A direct `DataDrivenEnsemble` useable for problems of the form `f(x,p,t,u) ↦ y`.

$(SIGNATURES)
"""
function DirectDataEnsemble(s::NamedTuple; name = gensym(:DDEnsemble), kwargs...)
    probs = map(keys(s)) do k
        si = s[k]
        _kwargs = collect_problem_kwargs(si; kwargs...)
        DataDrivenProblem(si[:X]; probtype = DDProbType(1), _kwargs...)
    end

    DataDrivenEnsemble(probs...; name = name)
end

"""
A time discrete `DataDrivenEnsemble` useable for problems of the form `f(x,p,t,u) ↦ x(t+1)`.

$(SIGNATURES)
"""
function DiscreteDataEnsemble(s::NamedTuple; name = gensym(:DDEnsemble), kwargs...)
    probs = map(keys(s)) do k
        si = s[k]
        _kwargs = collect_problem_kwargs(si; kwargs...)
        DataDrivenProblem(si[:X]; probtype = DDProbType(2), _kwargs...)
    end
    DataDrivenEnsemble(probs...; name = name)
end

"""
A time continuous `DataDrivenEnsemble` useable for problems of the form `f(x,p,t,u) ↦ dx/dt`.

$(SIGNATURES)

Automatically constructs derivatives via an additional collocation method, which can be either a collocation
or an interpolation from `DataInterpolations.jl` wrapped by an `InterpolationMethod` provided by the `collocation` keyworded argument.
"""
function ContinuousDataEnsemble(s::NamedTuple; name = gensym(:DDEnsemble), collocation = InterpolationMethod(), kwargs...)
    probs = map(keys(s)) do k
        si = s[k]
        # Check for differential states
        _kwargs = collect_problem_kwargs(si; kwargs...)
        if :DX ∈ keys(_kwargs)
            return DataDrivenProblem(si[:X]; probtype = DDProbType(3), _kwargs...)
        elseif :t ∈ keys(_kwargs)
            dx, x = collocate_data(si[:X], si[:t], collocation; kwargs...)
            return DataDrivenProblem(x; DX = dx, probtype = DDProbType(3), _kwargs...)
        else
            throw(ArgumentError("A continuous problem $(k) needs to have either derivative or time information specified!"))
        end
    end
    DataDrivenEnsemble(probs...; name = name)
end

Base.size(p::DataDrivenEnsemble) = size(p.probs)
Base.size(p::DataDrivenEnsemble, i) = size(p.probs[i]) 
Base.length(p::DataDrivenEnsemble) = length(p.probs)

function Base.summary(io::IO, x::DataDrivenEnsemble{N,C,P}) where {N,C,P}
    print(io, "$P DataDrivenEnsemble{$N} $(x.name) with $(length(x.probs)) problems")
    C ? nothing : print(io, " with controls")
    return
end

Base.show(io::IO, x::DataDrivenEnsemble{N,C,P}) where {N,C,P} = summary(io, x)

function Base.print(io::IO, x::DataDrivenEnsemble{N,C,P}) where {N,C,P}
    println(io, "$P DataDrivenEnsemble{$N} $(x.name)")
    println(io, "Summary")
    map(x.probs) do p
        print(io, p)
        println(io)
    end
    return
end
