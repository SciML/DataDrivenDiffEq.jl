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

function get_target(x::DataDrivenDataset)
    reduce(hcat, map(get_target, x.probs))
end

function init_implicits(x::DataDrivenDataset{N, W, C}) where {N, W, C}
    first_prob = first(x.probs)
    n_x, m = size(x)
    n_u = size(first_prob.U, 1)
    n_y = size(get_target(first_prob), 1)
    return (zeros(N, n_y + n_x, m),
            parameters(first_prob),
            zeros(N, m),
            n_u > 0 ? zeros(N, n_u, m) : N[])
end

function init_explicit(x::DataDrivenDataset{N, W, C}) where {N, W, C}
    first_prob = first(x.probs)
    n_x, m = size(x)
    n_u = size(first_prob.U, 1)
    n_y = size(get_target(first_prob), 1)
    return (zeros(N, n_x, m),
            parameters(first_prob),
            zeros(N, m),
            n_u > 0 ? zeros(N, n_u, m) : N[])
end

function get_oop_args(x::DataDrivenDataset{N, W, C}) where {N, W, C}
    X, p, t, U = init_explicit(x)
    last = 1
    @views for (i, s) in enumerate(cumsum(x.sizes))
        # Only copy if U is present
        if !W
            map(copyto!, (X[:, last:s], p, t[last:s], U[:, last:s]),
                get_oop_args(x.probs[i]))
        else
            map(copyto!, (X[:, last:s], p, t[last:s]), get_oop_args(x.probs[i])[1:3])
        end
        last += s
    end
    return (X, p, t, U)
end

function get_implicit_oop_args(x::DataDrivenDataset{N, W, C}) where {N, W, C}
    X, p, t, U = init_implicits(x)
    last = 1
    @views for (i, s) in enumerate(cumsum(x.sizes))
        # Only copy if U is present
        if !W
            map(copyto!, (X[:, last:s], p, t[last:s], U[:, last:s]),
                get_implicit_oop_args(x.probs[i]))
        else
            map(copyto!, (X[:, last:s], p, t[last:s]),
                get_implicit_oop_args(x.probs[i])[1:3])
        end
        last += s
    end
    return (X, p, t, U)
end

function (b::AbstractBasis)(d::DataDrivenDataset)
    reduce(hcat, map(b, d.probs))
end

function (b::AbstractBasis)(dx::AbstractMatrix, d::DataDrivenDataset)
    last = 1
    @views for (i, s) in enumerate(cumsum(d.sizes))
        b(dx[:, last:s], d.probs[i])
        last += s
    end
    return
end
