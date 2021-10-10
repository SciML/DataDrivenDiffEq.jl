## Coordinate transformations
# Just a collection of input mappings used for symmetries in the input data

# Abstract type
abstract type AbstractCoordinateTransform end

(c::AbstractCoordinateTransform)(x) = x
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
(c::AbstractCoordinateTransform)(x::AbstractMatrix) = reduce(hcat,map(c, eachcol(x)))

get_operator(x::CoordinateTransform{F}) where F = F
Base.length(x::CoordinateTransform) = length(x.inz)
Base.size(x::CoordinateTransform) = (length(x.inz),)

## Surrogates

abstract type AbstractDataDrivenSurrogate end

# Collect surrogate in a tree like structure

# Single linear surrogate
#mutable struct LinearSurrogate{T} <: AbstractSurrogate
#    """Coefficients"""
#    c::AbstractVector{T}
#    """Independent variables"""
#    vars::Vector{Int}
#    """Coordinate transformations"""
#    t::AbstractVector{AbstractCoordinateTransform}
#
#    function LinearSurrogate(c, inz)
#        return new{eltype(c)}(c, inz, [IdentityTransform()])
#    end
#
#    function LinearSurrogate(c, inz, t)
#        return new{eltype(c)}(c, inz, t)
#    end
#end
#
#(s::LinearSurrogate)(x) = s.c'x
#
#function Base.summary(io::IO, x::LinearSurrogate)
#    print(io, "Linear surrogate in $(sum(x.vars)) variables")
#    return
#end
#
#function Base.print(io::IO, x::LinearSurrogate)
#    summary(io, x)
#    print(io, "\nCoefficients : $(x.c)\n")
#    print(io, "Transformations : $(length(x.t))")
#end
#
#Base.show(io::IO, x::LinearSurrogate) = summary(io, x)

## Single nonlinear surrogate
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
        f, jac, inz, [IdentityTransform()], [identity]
    )

    # Explore input transformations
    h = explore_symmetries(f, x, inz, opts)

    # Explore output transformations

    # Explore compoisition
    _s = separate_function(f, x, inz, opts, depth = depth+1)

    isnothing(_s) &&  return DataDrivenSurrogate{linearity}(
        f, jac, inz, h, [identity]
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

## Utilities for applying functions (solve) to the surrogate

## Traversal 

#traverse(args...) = nothing
#
#traverse(x::AbstractSurrogate, i = 0) = begin
#    print(repeat("\t", i))
#    show(x)
#    println()
#end
#
#traverse(x::CompositeSurrogate{F}, i = 0) where F = begin 
#    print(repeat("\t", i))
#    show(x)
#    println()
#    traverse(x.left, i+1)
#    traverse(x.right, i+1)
#end
