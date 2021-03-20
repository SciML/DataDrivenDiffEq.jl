function _promote(args...)
    _type = Base.promote_eltype(args...)
    return map(x->convert.(_type, x), args)
end

_isfun(x) = false
_isfun(x::F) where F <: Function = true


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
struct DataDrivenProblem{dType} <: AbstractDataDrivenProblem where {dType <: Real}

    # Data
    """State measurements"""
    X::AbstractMatrix{dType}
    """Time measurements (optional)"""
    t::AbstractVector{dType}
    """Differental state measurements (optional) or measurements of the next state"""
    DX::AbstractMatrix{dType}
    """Output measurements (optional; not used right now)"""
    Y::AbstractMatrix{dType}
    """Input measurements (optional)"""
    U::AbstractMatrix{dType}


    """Parameters associated with the problem (optional)"""
    p::AbstractVector{dType}
    """(Time) discrete problem"""
    is_discrete::Bool

    function DataDrivenProblem(X, t, DX, Y, U, p, is_discrete)
        dType = Base.promote_eltype(X, t, DX, Y, U, p)
        return new{dType}(_promote(X,t,DX,Y,U,p)..., is_discrete)
    end


    function DataDrivenProblem(X, t, DX, Y, U::F, p, is_discrete) where F <: Function
        # Generate the input as a Matrix

        ts = isempty(t) ? zeros(eltype(X), size(X, 2)) : t

        u_ = hcat(map(i->U(X[:,i], p, ts[i]), 1:size(X,2))...)

        dType = Base.promote_eltype(X, t, DX, Y, u_, p)
        return new{dType}(_promote(X,t,DX,Y,u_,p)..., is_discrete)
    end
end


function DataDrivenProblem(X::AbstractMatrix;
    t::AbstractVector = Array{eltype(X)}(undef, 0),
    DX::AbstractMatrix = Array{eltype(X)}(undef, 0, 0),
    Y::AbstractMatrix = Array{eltype(X)}(undef, 0,0),
    U::F = Array{eltype(X)}(undef, 0,0),
    p::AbstractVector = Array{eltype(X)}(undef, 0),
    is_discrete::Bool = true) where F <: Union{AbstractMatrix, Function}


    return DataDrivenProblem(X,t,DX,Y,U,p,is_discrete)
end


## Discrete Constructors
"""
A time discrete `DataDrivenProblem`.

$(SIGNATURES)
"""
function DiscreteDataDrivenProblem(X::AbstractMatrix; kwargs...)
    DataDrivenProblem(X[:, 1:end-1], DX = X[:, 2:end], is_discrete = true; kwargs...)
end

function DiscreteDataDrivenProblem(X::AbstractMatrix, t::AbstractVector; kwargs...)
    DataDrivenProblem(X[:, 1:end-1], t=t, DX = X[:, 2:end], is_discrete = true; kwargs...)
end

function DiscreteDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, U_DX::AbstractMatrix; kwargs...)
    # We assume that if the size is equal, we have the next state measurements
    size(X, 1) == size(U_DX, 1) && return DataDrivenProblem(X, t=t, DX = U_DX, is_discrete = true; kwargs...)
    return DataDrivenProblem(X[:, 1:end-1], t=t, DX = X[:, 2:end], U = U_DX[:, 1:(size(X, 2)-1)], is_discrete = true; kwargs...)
end

function DiscreteDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, DX::AbstractMatrix, U::AbstractMatrix; kwargs...)
    return DataDrivenProblem(X, t=t, DX = DX, U = U, is_discrete = true; kwargs...)
end

function DiscreteDataDrivenProblem(X::AbstractMatrix, DX::AbstractMatrix, U::AbstractMatrix; kwargs...)
    return DataDrivenProblem(X, DX = DX, U = U, is_discrete = true; kwargs...)
end


function DiscreteDataDrivenProblem(X::AbstractMatrix, U::F; kwargs...) where F <: Function
    DataDrivenProblem(X[:, 1:end-1], DX = X[:, 2:end], p = p, U = U, is_discrete = true; kwargs...)
end

function DiscreteDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, U::F; kwargs...) where F <: Function
    DataDrivenProblem(X[:, 1:end-1], t=t, DX = X[:, 2:end], U = U, is_discrete = true; kwargs...)
end

function DiscreteDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, DX::AbstractMatrix, U::F; kwargs...) where F <: Function
    DataDrivenProblem(X, t=t, DX = DX, U = U, is_discrete = true; kwargs...)
end


## Continouos Constructors
"""
A time continuous `DataDrivenProblem`.

$(SIGNATURES)

Automatically constructs derivatives via an additional collocation method, which can be either a collocation
or an interpolation from `DataInterpolations.jl` wrapped by an `InterpolationMethod`.
"""
#ContinuousDataDrivenProblem(args...; kwargs...) = DataDrivenProblem(args...; kwargs...,  is_discrete = false)

function ContinuousDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, collocation = InterpolationMethod();kwargs...)
    dx, x = collocate_data(X, t, collocation)
    return DataDrivenProblem(x, t = t, DX = dx, is_discrete = false; kwargs...)
end

function ContinuousDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, U::F, collocation = InterpolationMethod(), kwargs...) where F <: Union{AbstractMatrix, Function}
    dx, x = collocate_data(X, t, collocation)
    return DataDrivenProblem(x, t = t, DX = dx, U = U, is_discrete = false; kwargs...)
end

function ContinuousDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, U_DX::AbstractMatrix; kwargs...)
    size(X) == size(U_DX) && DataDrivenProblem(X, t = t, DX = U_DX, is_discrete = false; kwargs...)
    ContinuousDataDrivenProblem(X, t, InterpolationMethod(), U = U_DX, is_discrete = false; kwargs...)
end

function ContinuousDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, DX::AbstractMatrix, U::F; kwargs...) where F <: Union{AbstractMatrix, Function}
    return DataDrivenProblem(X, t = t, DX = DX, U = U, is_discrete = false; kwargs...)
end

function ContinuousDataDrivenProblem(X::AbstractMatrix, DX::AbstractMatrix; kwargs...)
    return DataDrivenProblem(X, DX = DX, is_discrete = false; kwargs...)
end

function ContinuousDataDrivenProblem(X::AbstractMatrix, DX::AbstractMatrix, U::F; kwargs...) where F <: Union{AbstractMatrix, Function}
    return DataDrivenProblem(X, DX = DX, U = U, is_discrete = false; kwargs...)
end

## Utils

# Check for systemtype
is_discrete(x::DataDrivenProblem) = x.is_discrete
is_continuous(x::DataDrivenProblem) = !x.is_discrete

# Check for timepoints
has_timepoints(x::DataDrivenProblem) = !isempty(x.t)

# Check for inputs
has_inputs(x::DataDrivenProblem) = !isempty(x.U)

# Check for observations
has_observations(x::DataDrivenProblem) = _isfun(x.Y) ? true : !isempty(x.Y)

# Check for derivatives
has_derivatives(x::DataDrivenProblem) = !isempty(x.DX)

# Check for parameters
has_parameters(x::DataDrivenProblem) = !isempty(x.p)

# Check for nans, inf etc
function check_domain(x)
    isempty(x) && return false
    any(isnan.(x)) || any(isinf.(x))
end

# Check for validity

"""
$(SIGNATURES)

Checks if a `DataDrivenProblem` is valid by checking if the data contains `NaN`, `Inf` and
if the number of measurements is consistent.

# Example

```julia
is_valid(problem)
```
"""
function is_valid(x::DataDrivenProblem)
    # TODO Give a hint whats wrong here!

    # Check for nans, infs
    check_domain(x.X) && return false
    check_domain(x.DX) && return false
    # Checks only if 1 measurement is there
    size(x.X,2) != size(x.DX,2) && return false

    if is_discrete(x)
        # We assume to have measurements for all states here
        !isequal(x.X[:, 2:end], x.DX[:, 1:end-1]) && return false
    end

    if has_timepoints(x)
        length(x.t) != size(x.X, 2) && return false
    end

    if has_inputs(x) && isa(x.U, AbstractMatrix)
        size(x.X, 2) != size(x.U, 2) && return false
        check_domain(x.U) && return false
    end

    if has_observations(x)
        size(x.X, 2) != size(x.Y, 2) && return false
        check_domain(x.Y) && return false
    end

    if has_parameters(x)
        check_domain(x.p) && return false
    end


    return true
end

# TODO Can we optimize?
function get_oop_args(x::DataDrivenProblem)
    returns = []
    @inbounds for f in (:X, :p, :t, :U)
        x_ = getfield(x, f)
        push!(returns, x_)
    end
    return returns
end
