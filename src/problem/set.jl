"""
$(TYPEDEF)

A collection of DataDrivenProblems used to concatenate different trajectories or experiments.

Can be called with either a `NTuple` of problems or a `NamedTuple` of `NamedTuples`. 
Similar to the `DataDrivenProblem`, it has three constructors available:

+ `DirectDataset` for direct problems
+ `DiscreteDataset` for discrete problems
+ `ContinuousDataset` for continuous problems

# Fields
$(FIELDS)

# Signatures
$(SIGNATURES)

"""
struct DataDrivenDataset{N, U, C} <: AbstractDataDrivenProblem{N, U, C}
    """Name of the dataset"""
    name::Symbol
    """The problems"""
    probs::NTuple{M, AbstractDataDrivenProblem{N, U, C}} where {M}
    """The length of each problem - for internal use"""
    sizes::NTuple{M, Int} where {M}
end

# Constructor

function DataDrivenDataset(probs::Vararg{T, N}; name = gensym(:DDSet),
                           kwargs...) where {T <: AbstractDataDrivenProblem, N}
    return DataDrivenDataset(name, probs, map(length, probs))
end

function DataDrivenDataset(solutions::Vararg{T, N}; name = gensym(:DDSet),
                           kwargs...) where {T <: DiffEqBase.DESolution, N}
    probs = map(solutions) do s
        DataDrivenProblem(s; kwargs...)
    end
    return DataDrivenDataset(probs..., name = name)
end

"""
A direct `DataDrivenDataset` useable for problems of the form `f(x,p,t,u) ↦ y`.

$(SIGNATURES)
"""
function DirectDataset(s::NamedTuple; name = gensym(:DDSet), kwargs...)
    probs = map(keys(s)) do k
        si = s[k]
        _kwargs = collect_problem_kwargs(si; kwargs...)
        DataDrivenProblem(si[:X]; probtype = DDProbType(1), _kwargs...)
    end

    DataDrivenDataset(probs...; name = name)
end

"""
A time discrete `DataDrivenDataset` useable for problems of the form `f(x,p,t,u) ↦ x(t+1)`.

$(SIGNATURES)
"""
function DiscreteDataset(s::NamedTuple; name = gensym(:DDSet), kwargs...)
    probs = map(keys(s)) do k
        si = s[k]
        _kwargs = collect_problem_kwargs(si; kwargs...)
        DataDrivenProblem(si[:X]; probtype = DDProbType(2), _kwargs...)
    end
    DataDrivenDataset(probs...; name = name)
end

"""
A time continuous `DataDrivenDataset` useable for problems of the form `f(x,p,t,u) ↦ dx/dt`.

$(SIGNATURES)

Automatically constructs derivatives via an additional collocation method, which can be either a collocation
or an interpolation from `DataInterpolations.jl` wrapped by an `InterpolationMethod` provided by the `collocation` keyworded argument.
"""
function ContinuousDataset(s::NamedTuple; name = gensym(:DDSet),
                           collocation = InterpolationMethod(), kwargs...)
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
    DataDrivenDataset(probs...; name = name)
end

collect_problem_kwargs(s; kwargs...) = begin
    _kwargs = Dict()
    for k in keys(s)
        if k ∈ [:DX, :t, :Y, :U, :p] # Very specific subset
            _kwargs[k] = s[k]
        end
    end
    merge(_kwargs, kwargs)
end

Base.length(s::DataDrivenDataset) = sum(s.sizes)
Base.size(s::DataDrivenDataset) = (first(size(first(s.probs))), length(s))

function Base.summary(io::IO, x::DataDrivenDataset{N, C, P}) where {N, C, P}
    print(io, "$P Dataset{$N} $(x.name) with $(length(x.probs)) problems")
    n, m = size(x)
    print(io, " in $n dimensions and $m samples")
    C ? nothing : print(io, " with controls")
    return
end

Base.show(io::IO, x::DataDrivenDataset{N, C, P}) where {N, C, P} = summary(io, x)

function Base.print(io::IO, x::DataDrivenDataset{N, C, P}) where {N, C, P}
    println(io, "$P DataDrivenDataset{$N} $(x.name)")
    println(io, "Summary")
    map(x.probs) do p
        print(io, p)
        println(io)
    end
    return
end

function is_valid(x::DataDrivenDataset)
    all(map(is_valid, x.probs))
end

function get_implicit_data(x::DataDrivenDataset)
    reduce(hcat, map(get_implicit_data, x.probs))
end

# We assume common parameters (for now)
function ModelingToolkit.parameters(x::DataDrivenDataset, i = :)
    parameters(first(x.probs), i)
end

function get_oop_args(x::DataDrivenDataset)
    data = map(x.probs) do p
        get_oop_args(p)
    end
    X, p, t, U = first(data)
    for d in Base.tail(data)
        X̂, _, t̂, Û = d
        X = hcat(X, X̂)
        t = vcat(t, t̂)
        U = hcat(U, Û)
    end
    X, p, t, U
end
