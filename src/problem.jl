function _promote(args...)
    _type = Base.promote_eltype(args...)
    return map(x->convert.(_type, x), args)
end

_isfun(x) = false
_isfun(x::F) where F <: Function = true

abstract type AbstractDataDrivenProblem end

"""
$(TYPEDEF)

The `DataDrivenProblem` defines a general estimation problem given measurements, inputs and (in the near future) observations.
Two construction methods are available:

+ `DiscreteDataDrivenProblem` for time discrete systems
+ `ContinousDataDrivenProblem` for systems continouos in time

both are aliases for constructing a problem.

# Fields
$(FIELDS)

# Signatures
$(SIGNATURES)

# Example

```julia
X, DX, t = data...

# Define a discrete time problem
prob = DiscreteDataDrivenProblem(X)

# Define a continous time problem without explicit time points
prob = ContinuousDataDrivenProblem(X, DX)

# Define a continous time problem without explict derivatives
prob = ContinuousDataDrivenProblem(X, t)

# Define a discrete time problem with an input function as a function
input_signal(u,p,t) = t^2
prob = DiscreteDataDrivenProblem(X, t, input_signal)
```
"""
struct DataDrivenProblem{dType, uType} <: AbstractDataDrivenProblem where {dType <: Real, uType <: Union{AbstractMatrix, Function}}

    # Data
    """State measurements"""
    X::AbstractMatrix{dType}
    """Time measurements (optional)"""
    t::AbstractVector{dType}
    """Differental state measurements (optional)"""
    DX::AbstractMatrix{dType}
    """Output measurements (optional; not used right now)"""
    Y::AbstractMatrix{dType}
    """Input measurements (optional) provided either as an `AbstractMatrix` or a `Function` of form `f(u,p,t)` which returns an `AbstractVector`"""
    U::uType

    """(Time) discrete problem"""
    is_discrete::Bool

    function DataDrivenProblem(X, t, DX, Y, U::F, is_discrete) where F <: AbstractMatrix
        dType = Base.promote_eltype(X, t, DX, Y, U)
        return new{dType, typeof(U)}(_promote(X,t,DX,Y,U)..., is_discrete)
    end


    function DataDrivenProblem(X, t, DX, Y, U::F, is_discrete) where F <: Function
        dType = Base.promote_eltype(X, t, DX, Y)
        return new{dType, typeof(U)}(_promote(X,t,DX,Y)..., U, is_discrete)
    end
end


function DataDrivenProblem(X::AbstractMatrix;
    t::AbstractVector = Array{eltype(X)}(undef, 0),
    DX::AbstractMatrix = Array{eltype(X)}(undef, 0, 0),
    Y::AbstractMatrix = Array{eltype(X)}(undef, 0,0),
    U::F = Array{eltype(X)}(undef, 0,0),
    is_discrete::Bool = true) where F <: Union{AbstractMatrix, Function}

    return DataDrivenProblem(X,t,DX,Y,U, is_discrete)
end

## Discrete Constructors
"""
A time discrete `DataDrivenProblem`.

$(SIGNATURES)
"""
DiscreteDataDrivenProblem(args...; kwargs...) = DataDrivenProblem(args...;kwargs..., is_discrete = true)
DiscreteDataDrivenProblem(X::AbstractMatrix, t::AbstractVector) =  DiscreteDataDrivenProblem(X, t=t)
DiscreteDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, U::F) where F <: Union{AbstractMatrix, Function} =  DiscreteDataDrivenProblem(X, t=t, U = U)
DiscreteDataDrivenProblem(X::AbstractMatrix, U::F) where F <: Union{AbstractMatrix, Function} =  DiscreteDataDrivenProblem(X, U = U)


## Continouos Constructors
"""
A time continuous `DataDrivenProblem`.

$(SIGNATURES)

Automatically constructs derivatives via an additional collocation method, which can be either:

or a wrapped interpolation from `DataInterpolations.jl`
"""
ContinuousDataDrivenProblem(args...; kwargs...) = DataDrivenProblem(args...; kwargs...,  is_discrete = false)

function ContinuousDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, collocation = InterpolationMethod())
    dx, x = collocate_data(X, t, collocation)
    return ContinuousDataDrivenProblem(x, t = t, DX = dx)
end

function ContinuousDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, U::F, collocation = InterpolationMethod()) where F <: Union{AbstractMatrix, Function}
    dx, x = collocate_data(X, t, collocation)
    return ContinuousDataDrivenProblem(x, t = t, DX = dx, U = U)
end

function ContinuousDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, U_DX::AbstractMatrix)
    size(X) == size(U_DX) ? ContinuousDataDrivenProblem(X, t = t, DX = U_DX) : ContinuousDataDrivenProblem(X, t, U, InterpolationMethod())
end

function ContinuousDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, DX::AbstractMatrix, U::F) where F <: Union{AbstractMatrix, Function}
    return ContinuousDataDrivenProblem(X, t = t, DX = DX, U = U)
end

## Utils

# Check for systemtype
is_discrete(x::DataDrivenProblem) = x.is_discrete
is_continuous(x::DataDrivenProblem) = !x.is_discrete

# Check for timepoints
has_timepoints(x::DataDrivenProblem) = !isempty(x.t)

# Check for inputs
has_inputs(x::DataDrivenProblem) = _isfun(x.U) ? true : !isempty(x.U)

# Check for observations
has_observations(x::DataDrivenProblem) = _isfun(x.Y) ? true : !isempty(x.Y)

# Check for derivatives
has_derivatives(x::DataDrivenProblem) = !isempty(x.DX)

# Check for nans, inf etc
check_domain(x) =  any(isnan.(x) || isinf.(x))

# Check for validity

"""
$(METHODS)

Checks if a `DataDrivenProblem` is valid by checking if the data contains `NaN`, `Inf` and
if the number of measurements is consistent.

# Example

```julia
is_valid(problem)
```
"""
function is_valid(x::DataDrivenProblem)
    # Check for nans, infs
    check_domain(x.X) && return false

    if has_timepoints(x)
        length(x.t) != size(x.X, 2) && return false
    end

    if has_derivatives(x)
        size(x.X) != size(x.DX) && return false
        check_domain(x.DX) && return false

    end

    if has_inputs(x) && isa(x.U, AbstractMatrix)
        size(x.X, 2) != size(x.U, 2) && return false
        check_domain(x.U) && return false
    end

    if has_observations(x)
        size(x.X, 2) != size(x.Y, 2) && return false
        check_domain(x.Y) && return false
    end

    return true
end
