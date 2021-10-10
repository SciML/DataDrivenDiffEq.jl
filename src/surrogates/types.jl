## Coordinate transformations
# Just a collection of input mappings used for symmetries in the input data

# Abstract type
abstract type AbstractCoordinateTransform end

(c::AbstractCoordinateTransform)(x) = x
(c::AbstractCoordinateTransform)(x::AbstractMatrix) = hcat(map(c, eachcol(x))...)

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
(c::AbstractCoordinateTransform)(x::AbstractMatrix) = hcat(map(c, eachcol(x))...)

get_operator(x::CoordinateTransform{F}) where F = F
len_inz(x::CoordinateTransform) = length(x.inz)


## Surrogates

abstract type AbstractSurrogate end

# Collect surrogate in a tree like structure

# Single linear surrogate
mutable struct LinearSurrogate{T} <: AbstractSurrogate
    """Coefficients"""
    c::AbstractVector{T}
    """Independent variables"""
    vars::Vector{Int}
    """Coordinate transformations"""
    t::AbstractVector{AbstractCoordinateTransform}

    function LinearSurrogate(c, inz)
        return new{eltype(c)}(c, inz, [IdentityTransform()])
    end

    function LinearSurrogate(c, inz, t)
        return new{eltype(c)}(c, inz, t)
    end
end

(s::LinearSurrogate)(x) = s.c'x

function Base.summary(io::IO, x::LinearSurrogate)
    print(io, "Linear surrogate in $(sum(x.vars)) variables")
    return
end

function Base.print(io::IO, x::LinearSurrogate)
    summary(io, x)
    print(io, "\nCoefficients : $(x.c)\n")
    print(io, "Transformations : $(length(x.t))")
end

Base.show(io::IO, x::LinearSurrogate) = summary(io, x)

## Single nonlinear surrogate
mutable struct NonlinearSurrogate <: AbstractSurrogate
    """Surrogate function"""
    f::Function
    """Jacobian function"""
    j::Function
    """Independent variables"""
    vars::Vector{Int}
    """Coordinate transformations"""
    t::AbstractVector{AbstractCoordinateTransform}

    function NonlinearSurrogate(f, inz; 
        j = _gradient(f), transforms = [IdentityTransform()])
        return new(f, j, inz, transforms)
    end
end

(s::NonlinearSurrogate)(x) = s.f(x)

function Base.summary(io::IO, x::NonlinearSurrogate)
    print(io, "Nonlinear surrogate in $(sum(x.vars)) variables")
    return
end

function Base.print(io::IO, x::NonlinearSurrogate)
    summary(io, x)
    print(io, "\nTransformations : $(length(x.t))")
end

Base.show(io::IO, x::NonlinearSurrogate) = summary(io, x)

# Binary tree
# F is the compoisition, left and right the different surrogates at the
# next level
mutable struct CompositeSurrogate{F} <: AbstractSurrogate
    left::AbstractSurrogate
    right::AbstractSurrogate

    function CompositeSurrogate(op, left, right)
        return new{op}(left, right)
    end
end

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

## Traversal 

traverse(args...) = nothing

traverse(x::AbstractSurrogate, i = 0) = begin
    print(repeat("\t", i))
    show(x)
    println()
end

traverse(x::CompositeSurrogate{F}, i = 0) where F = begin 
    print(repeat("\t", i))
    show(x)
    println()
    traverse(x.left, i+1)
    traverse(x.right, i+1)
end
