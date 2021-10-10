## Coordinate transformations
# Just a collection of input mappings used for symmetries in the input data

(c::AbstractCoordinateTransform)(x::T) where T <: Number = x
(c::AbstractCoordinateTransform)(x::AbstractVector) = x
(c::AbstractCoordinateTransform)(x::AbstractMatrix) = reduce(hcat,map(c, eachcol(x)))

# IdentityTransform as a special type
struct IdentityTransform <: AbstractCoordinateTransform end

(c::IdentityTransform)(x::AbstractMatrix) = x


"""
Single coordinate transformation. Inz indicates the coordinates to use and F is a function 
performing the mmaping.
"""
struct CoordinateTransform{F} <: AbstractCoordinateTransform
    inz::Vector{Int}
    function CoordinateTransform(f, inz)
        return new{f}(inz)
    end
end

(c::CoordinateTransform{F})(x::AbstractVector) where {F} = F(getindex(x, c.inz)...)

get_operator(x::CoordinateTransform{F}) where F = F
Base.length(x::CoordinateTransform) = length(x.inz)
Base.size(x::CoordinateTransform) = (length(x.inz),)

## Surrogates

# Single nonlinear surrogate
mutable struct DataDrivenSurrogate{Linearity} <: AbstractDataDrivenSurrogate
    """Surrogate function"""
    f::Function
    
    """Jacobian function"""
    j::Function
    
    """Independent variables"""
    vars::Vector{Int}

    """Input coordinate transformations"""
    h::AbstractVector{AbstractCoordinateTransform}

    """Output coordinate transformation"""
    g::AbstractVector{Function}
end

function DataDrivenSurrogate(f, x, 
    opts = InvestigationOptions(f(x), f, x), 
    depth = 0)

    depth >= opts.kwargs["max_depth"] && return nothing

    # Create a datadriven surrogate
    y = f(x[:,1])
    
    # If function has a vector output, return one surrogate for 
    # each output

    if !(typeof(y) <: Number) 
        _s = map(1:size(y, 1)) do i
            _f = _create_f(f, i)
           return DataDrivenSurrogate(_f, x)
        end      
        return _s
    end

    # Create jacobian
    jac = _gradient(f)

    # Create inzidenz
    inz = create_incidence(f, x, opts, jac)

    # Check linearity
    linearity = is_linear(f, x, opts, jac)
    
    linearity && return DataDrivenSurrogate{linearity}(
        f, jac, inz, AbstractCoordinateTransform[], Function[]
    )

    # Explore input transformations
    h = explore_symmetries(f, x, inz, opts)

    # Explore output transformations

    # Explore compoisition
    _s = separate_function(f, x, inz, opts, depth = depth+1)

    isnothing(_s) &&  return DataDrivenSurrogate{linearity}(
        f, jac, inz, h, Function[]
    )

    return _s
end

"""
Original function call.
"""
(s::DataDrivenSurrogate)(x) = s.f(x)

function Base.summary(io::IO, x::DataDrivenSurrogate{true})
    print(io, "Linear surrogate in $(sum(x.vars)) variables")
    return
end

function Base.summary(io::IO, x::DataDrivenSurrogate{F} where F)
    print(io, "Nonlinear surrogate in $(sum(x.vars)) variables")
    return
end

function Base.print(io::IO, x::DataDrivenSurrogate)
    summary(io, x)
    print(io, "\nInzidence : $(x.vars)")
    print(io, "\nTransformations : $(length(x.h))")
end

Base.show(io::IO, x::DataDrivenSurrogate) = summary(io, x)

# Binary tree
# F is the compoisition, left and right the different surrogates at the
# next level
mutable struct CompositeSurrogate{F} <: AbstractDataDrivenSurrogate
    left::AbstractDataDrivenSurrogate
    right::AbstractDataDrivenSurrogate

    function CompositeSurrogate(op, left, right)
        return new{op}(left, right)
    end
end

get_operator(s::CompositeSurrogate{F}) where F = F
get_operator(s::AbstractDataDrivenSurrogate) = identity


(s::CompositeSurrogate{F})(x) where F = broadcast(F, s.left(x), s.right(x))
left(s::CompositeSurrogate) = s.left
right(s::CompositeSurrogate) = s.right
children(s::CompositeSurrogate) = (s.left, s.right)

function Base.summary(io::IO, x::CompositeSurrogate{F}) where F
    print(io, "Composite surrogate with operation $F")
    return
end

function Base.print(io::IO, x::CompositeSurrogate)
    summary(io, x)
    print(io, "\nLeft child :\n")
    print(io, left(x), "\n")
    print(io, "\nRight child :\n")
    print(io, right(x))
end

Base.show(io::IO, x::CompositeSurrogate) = summary(io, x)


## Utilities
depth(x::AbstractDataDrivenSurrogate) = 0
depth(x::DataDrivenSurrogate) = 1
depth(x::CompositeSurrogate) = 1 + max(depth(x.right), depth(x.left))

get_operator(x::AbstractDataDrivenSurrogate) = identity
get_operator(x::CompositeSurrogate{F}) where F = F

apply_function_tree(f, s::AbstractDataDrivenSurrogate, args...; kwargs...) = f(s, args...; kwargs...)
function apply_function_tree(f, s::CompositeSurrogate, args...; kwargs...)
    return map(x->apply_function_tree(f, x, args...; kwargs...), children(s))
end

is_linear(s::AbstractDataDrivenSurrogate) = false
is_linear(s::DataDrivenSurrogate{true}) = true
is_linear(s::CompositeSurrogate{+}) = all(map(is_linear, children(s)))

input_transformations(s::AbstractDataDrivenSurrogate) = AbstractCoordinateTransform[]
input_transformations(s::DataDrivenSurrogate) = getfield(s, :h)
input_transformations(s::CompositeSurrogate) = map(input_transformations, children(s))