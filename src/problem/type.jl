function _promote(args...)
    _type = Base.promote_eltype(args...)
    return map(x->convert.(_type, x), args)
end

## Utilities

"""
$(SIGNATURES)

Check if the problem has control inputs.
"""
is_autonomous(::AbstractDataDrivenProblem{N, U, C}) where {N,U,C} = U


"""
$(SIGNATURES)

Check if the problem is time discrete.
"""
is_discrete(::AbstractDataDrivenProblem{N, U, C}) where {N,U,C} = C == DDProbType(2)

"""
$(SIGNATURES)

Check if the problem is direct.
"""
is_direct(::AbstractDataDrivenProblem{N, U, C}) where {N,U,C} = C == DDProbType(1)

"""
$(SIGNATURES)

Check if the problem is time continuous.
"""
is_continuous(::AbstractDataDrivenProblem{N, U, C}) where {N,U,C} = C == DDProbType(3)

"""
$(SIGNATURES)

Check if the problem is parameterized.
"""
is_parametrized(x::AbstractDataDrivenProblem{N, U, C}) where {N,U,C} = hasfield(typeof(x), :p) && !isempty(x.p)

"""
$(SIGNATURES)

Check if the problem has associated measurement times.
"""
has_timepoints(x::AbstractDataDrivenProblem{N,U,C}) where {N,U,C} = !isempty(x.t)

## Concrete Type
"""
$(TYPEDEF)

The `DataDrivenProblem` defines a general estimation problem given measurements, inputs and (in the near future) observations.
Two construction methods are available:

+ `DiscreteDataDrivenProblem` for time discrete systems
+ `ContinousDataDrivenProblem` for systems continuous in time

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

# Define a continuous time problem without explicit time points
prob = ContinuousDataDrivenProblem(X, DX)

# Define a continuous time problem without explicit derivatives
prob = ContinuousDataDrivenProblem(X, t)

# Define a discrete time problem with an input function as a function
input_signal(u,p,t) = t^2
prob = DiscreteDataDrivenProblem(X, t, input_signal)
```
"""
struct DataDrivenProblem{dType, cType, probType} <: AbstractDataDrivenProblem{dType, cType, probType}

    # Data
    """State measurements"""
    X::AbstractMatrix{dType}
    """Time measurements (optional)"""
    t::AbstractVector{dType}

    """Differental state measurements (optional); Used for time continuous problems"""
    DX::AbstractMatrix{dType}
    """Output measurements (optional); Used for direct problems"""
    Y::AbstractMatrix{dType}
    """Input measurements (optional); Used for non-autonoumous problems"""
    U::AbstractMatrix{dType}

    """Parameters associated with the problem (optional)"""
    p::AbstractVector{dType}

    """Name of the problem"""
    name::String
end


function DataDrivenProblem(probType, X, t, DX, Y, U, p; name = "", kwargs...)
    dType = Base.promote_eltype(X, t, DX, Y, U, p)
    cType = isempty(U)
    
    # We assume a discrete Problem
    if isnothing(probType) 
        probType = DDProbType(2)
        if (isempty(DX) && !isempty(Y))
            probType = DDProbType(1) # Direct problem
        elseif !isempty(DX)
            probType = DDProbType(3) # Continouos
        end
    end

    return DataDrivenProblem{dType, cType, probType}(_promote(X,t,DX,Y,U,p)..., name)
end


function DataDrivenProblem(probtype, X, t, DX, Y, U::F, p; kwargs...) where F <: Function
    # Generate the input as a Matrix

    ts = isempty(t) ? zeros(eltype(X),size(X,2)) : t

    u_ = hcat(map(i->U(X[:,i], p, ts[i]), 1:size(X,2))...)

    return DataDrivenProblem(probtype, _promote(X,t,DX,Y,u_,p)...; kwargs...)
end


function DataDrivenProblem(X::AbstractMatrix;
    t::AbstractVector = collect(one(eltype(X)):size(X,2)),
    DX::AbstractMatrix = Array{eltype(X)}(undef, 0, 0),
    Y::AbstractMatrix = Array{eltype(X)}(undef, 0,0),
    U::F = Array{eltype(X)}(undef, 0,0),
    p::AbstractVector = Array{eltype(X)}(undef, 0),
    probtype = nothing,
    kwargs...) where F <: Union{AbstractMatrix, Function}

    return DataDrivenProblem(probtype, X,t,DX,Y,U,p; kwargs...)
end

function Base.summary(io::IO, x::DataDrivenProblem{N,C,P}) where {N,C,P}
    print(io, "$P DataDrivenProblem{$N}")
    isempty(x.name) ? nothing : print(io, " $(x.name)")
    n,m = size(x.X)
    print(io, " in $n dimensions and $m samples")
    C ? nothing : print(io, " with controls")
    return
end

function Base.print(io::IO, x::AbstractDataDrivenProblem{N,C,P}) where {N,C,P}
    isempty(x.name) ? nothing : println(io, "$(x.name)")
    println(io, "$P DataDrivenProblem{$N}")
    println(io, "Summary")
    n,m = size(x.X)
    println(io, "$m measurements")
    println(io, "$n state(s)")
    n = size(x.DX, 1)
    isempty(x.DX) ? nothing : println("$n differential state(s)")
    n = size(x.Y, 1)
    isempty(x.Y) ? nothing : println("$n observed variable(s)")
    n = size(x.U, 1)
    isempty(x.U) ? nothing : println(io, "$n control(s)")
end

Base.show(io::IO, x::DataDrivenProblem{N,C,P}) where {N,C,P} = summary(io, x)


## Discrete Constructors
"""
A time discrete `DataDrivenProblem` useable for problems of the form `f(x[i],p,t,u) ↦ x[i+1]`.

$(SIGNATURES)
"""
function DiscreteDataDrivenProblem(X::AbstractMatrix; kwargs...)
    DataDrivenProblem(X; probtype = DDProbType(2), kwargs...)
end

function DiscreteDataDrivenProblem(X::AbstractMatrix, t::AbstractVector; kwargs...)
    DataDrivenProblem(X; t=t,  probtype = DDProbType(2),  kwargs...)
end

function DiscreteDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, U::AbstractMatrix; kwargs...)
    return DataDrivenProblem(X; t=t, U = U , probtype = DDProbType(2),  kwargs...)
end

function DiscreteDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, U::Function; kwargs...)
    return DataDrivenProblem(X; t=t, U = U ,  probtype = DDProbType(2), kwargs...)
end

## Continouos Constructors
"""
A time continuous `DataDrivenProblem` useable for problems of the form `f(x,p,t,u) ↦ dx/dt`.

$(SIGNATURES)

Automatically constructs derivatives via an additional collocation method, which can be either a collocation
or an interpolation from `DataInterpolations.jl` wrapped by an `InterpolationMethod`.
"""
function ContinuousDataDrivenProblem(X::AbstractMatrix, DX::AbstractMatrix; kwargs...)
    return DataDrivenProblem(X; DX = DX,  probtype = DDProbType(3), kwargs...)
end

function ContinuousDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, DX::AbstractMatrix; kwargs...)
    return DataDrivenProblem(X; t = t, DX = DX, probtype = DDProbType(3), kwargs...)
end

function ContinuousDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, DX::AbstractMatrix, U::AbstractMatrix; kwargs...)
    return DataDrivenProblem(X; t = t, DX = DX, U = U, probtype = DDProbType(3), kwargs...)
end


function ContinuousDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, DX::AbstractMatrix, U::F; kwargs...) where {F <: Function}
    return DataDrivenProblem(X; t = t, DX = DX, U = U, probtype = DDProbType(3), kwargs...)
end

function ContinuousDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, collocation = InterpolationMethod(); kwargs...)
    dx, x = collocate_data(X, t, collocation)
    return DataDrivenProblem(x; t = t, DX = dx, probtype = DDProbType(3), kwargs...)
end

function ContinuousDataDrivenProblem(X::AbstractMatrix, t::AbstractVector,  U::AbstractMatrix, collocation; kwargs...)
    dx, x = collocate_data(X, t, collocation)
    return DataDrivenProblem(x; t = t, DX = dx, U = U, probtype = DDProbType(3), kwargs...)
end

## Direct Constructors
"""
A direct `DataDrivenProblem` useable for problems of the form `f(x,p,t,u) ↦ y`.

$(SIGNATURES)
"""
function DirectDataDrivenProblem(X::AbstractMatrix, Y::AbstractMatrix; kwargs...)
    return DataDrivenProblem(X; Y = Y, probtype = DDProbType(1), kwargs...)
end

function DirectDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, Y::AbstractMatrix; kwargs...)
    return DataDrivenProblem(X; t = t, Y = Y, probtype = DDProbType(1), kwargs...)
end

function DirectDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, Y::AbstractMatrix, U; kwargs...)
    return DataDrivenProblem(X; t = t, Y = Y, U = U, probtype = DDProbType(1),kwargs...)
end

Base.length(prob::AbstractDataDrivenProblem) = size(prob.X, 2)
Base.size(prob::AbstractDataDrivenProblem) = size(prob.X)

"""
$(SIGNATURES)

Returns the name of the `problem`.
"""
get_name(p::AbstractDataDrivenProblem) = getfield(p, :name)

## Utils

function ModelingToolkit.states(p::AbstractDataDrivenProblem, i = :, j = :)
    x = getfield(p, :X)
    isempty(x) ? x : getindex(x, i, j)
end

function ModelingToolkit.parameters(p::AbstractDataDrivenProblem, i = :)
    x = getfield(p, :p)
    isempty(x) ? x : getindex(x, i)
end

function ModelingToolkit.independent_variable(p::AbstractDataDrivenProblem, i = :)
    x = getfield(p, :t)
    isempty(x) ? x : getindex(x, i)
end

function ModelingToolkit.get_dvs(p::AbstracContProb, i = :, j = :)
    x = getfield(p, :DX)
    isempty(x) ? x : getindex(x, i, j)
end

function ModelingToolkit.get_dvs(p::AbstractDirectProb, i = :, j = :)
    x = getfield(p, :Y)
    isempty(x) ? x : getindex(x, i, j)
end

function ModelingToolkit.get_dvs(p::AbstractDiscreteProb, i = :, j = :)
    ModelingToolkit.states(p, i, j)
end

function ModelingToolkit.observed(p::AbstractDataDrivenProblem, i = :, j = :)
    x = getfield(p, :Y)
    isempty(x) ? x : getindex(x, i, j)
end

function ModelingToolkit.controls(p::AbstractDataDrivenProblem, i = :, j = :)
    x = getfield(p, :U)
    isempty(x) ? x : getindex(x, i, j)
end



function Base.getindex(p::AbstractDataDrivenProblem, i = :, j = :)
    return (
        ModelingToolkit.states(p, i, j), 
        ModelingToolkit.parameters(p),
        ModelingToolkit.independent_variable(p, j),
        ModelingToolkit.controls(p, i, j)
    )  
end


# TODO This is just for explicit basis! 
# Make the basis callable with the problem
# Explicit


@views (b::AbstractBasis)(p::AbstractDataDrivenProblem) = begin 
    b(states(p), parameters(p), independent_variable(p), controls(p))
end

@views (b::AbstractBasis)(dx::AbstractMatrix, p::AbstractDataDrivenProblem) = begin
    b(dx, states(p), parameters(p), independent_variable(p), controls(p))
end

@views (b::AbstractBasis)(p::AbstractDataDrivenProblem, j) = begin 
    b(p[:, j]...)
end

@views (b::AbstractBasis)(dx::AbstractMatrix, p::AbstractDataDrivenProblem, j) = begin
    b(dx, p[:, j]...)
end

# Special case discrete problem

@views (b::AbstractBasis)(p::AbstractDiscreteProb) = begin 
    b(p[:, 1:length(p)-1]...)
end

@views (b::AbstractBasis)(dx::AbstractMatrix, p::AbstractDiscreteProb) = begin 
    b(dx, p, 1:length(p)-1)
end

# Check for nans, inf etc
check_domain(x) =  @assert all(.~isnan.(x)) && all(.~isinf.(x)) ("One or more measurements contain `NaN` or `Inf`.")
check_lengths(args...) = @assert all(map(x->size(x)[end], args) .== size(first(args))[end]) "One or more measurements are not sized equally."

# Return the target variables
get_target(x::AbstractDirectProb{N,C}) where {N,C} = x.Y
get_target(x::AbstractDiscreteProb{N,C}) where {N,C} = x.X[:,2:end]
get_target(x::AbstracContProb{N,C}) where {N,C} = x.DX


get_oop_args(x::AbstractDataDrivenProblem{N,C,P}) where {N <: Number, C, P} = map(f->getfield(x, f), (:X, :p, :t, :U))


function get_oop_args(x::AbstractDiscreteProb{N,C}) where {N <: Number, C}
    return  (
        x.X[:, 1:end-1],
        x.p,
        x.t[1:end-1],
        x.U[:, 1:end-1]
    )
end

function get_implicit_oop_args(x::AbstractDirectProb{N,C}) where {N <: Number, C}
    return  (
        [x.X; x.Y],
        x.p,
        x.t,
        x.U
    )
end

function get_implicit_oop_args(x::AbstracContProb{N,C}) where {N <: Number, C}
    return  (
        [x.X; x.DX],
        x.p,
        x.t,
        x.U
    )
end


function get_implicit_oop_args(x::AbstractDiscreteProb{N,C}) where {N <: Number, C}
    return  (
        [x.X[:, 1:end-1]; x.X[:, 2:end]],
        x.p,
        x.t[1:end-1],
        x.U[:, 1:end-1]
    )
end


"""
$(SIGNATURES)

Checks if a `DataDrivenProblem` is valid by checking if the data contains `NaN`, `Inf` and
if the number of measurements is consistent.

# Example

```julia
is_valid(problem)
```
"""
function is_valid(x::AbstractDirectProb{N,C}) where {N <: Number,C}
    map(check_domain, (x.X, x.Y, x.U, x.t, x.p))
    if !C && !isempty(x.t)
        check_lengths(x.X, x.Y, x.U, x.t)
    elseif !C
        check_lengths(x.X, x.Y, x.U)
    elseif C && !isempty(x.t)
        check_lengths(x.X, x.Y, x.t)
    else
        check_lengths(x.X, x.Y)
    end
    return true
end


function is_valid(x::AbstractDiscreteProb{N,C}) where {N <: Number,C}
    map(check_domain, (x.X, x.U, x.t, x.p))
    if !C && !isempty(x.t)
        check_lengths(x.X, x.U, x.t)
    elseif !C
        check_lengths(x.X, x.U)
    elseif C && !isempty(x.t)
        check_lengths(x.X, x.t)
    else
        check_lengths(x.X)
    end
    return true
end

function is_valid(x::AbstracContProb{N,C}) where {N <: Number,C}
    map(check_domain, (x.X, x.DX, x.U, x.t, x.p))
    if !C && !isempty(x.t)
        check_lengths(x.X, x.DX, x.U, x.t)
    elseif !C
        check_lengths(x.X, x.DX, x.U)
    elseif C && !isempty(x.t)
        check_lengths(x.X, x.DX, x.t)
    else
        check_lengths(x.X, x.DX)
    end
    return true
end

"""
$(SIGNATURES)

Asserts if the given combination of `problem` and `basis` - and optionally a target matrix `dx` for in place evaluation - is
valid in its dimensions. Should only be used for checking explicit evaluation. Can also be used as a shorthand to 

```julia 
@is_applicable problem # Checks if the problem is well posed in terms of dimensions
@is_applicable problem basis # Checks if the basis can be called with the problem out of place
@is_applicable problem basis dx # Checks if the basis can be called with the problem and matrix dx in place
```
"""
macro is_applicable(problem)
    return :(@assert is_valid($(esc(problem))))
end


macro is_applicable(problem, basis)
    ex = quote @assert length(states($(esc(basis)))) == size(states($(esc(problem))),1) "Problem and basis need to have same state size"
    @assert length(controls($(esc(basis)))) == size(controls($(esc(problem))),1) "Problem and basis need to have same control size"
    @assert length(parameters($(esc(basis)))) <= length(parameters($(esc(problem)))) "Problem and basis need to have consistent parameter size"
    end
    return ex
end

macro is_applicable(problem, basis, dx)
    return quote
        @is_applicable $(esc(problem)) $(esc(basis))
        lp = length($(esc(problem)))
        n, m = size($(esc(dx)))
        lb = length($(esc(basis)))
        if isa($(esc(problem)), AbstractDiscreteProb)
            @assert n == lb && m == lp-1 "Target array has to be of size ($lb, $lp)"
        else
            @assert n == lb && m == lp "Target array has to be of size ($lb, $lp)"
        end
    end
end


## DESolution dispatch

ContinuousDataDrivenProblem(sol::T; kwargs...) where T <: DiffEqBase.DESolution = DataDrivenProblem(sol; kwargs...)
DiscreteDataDrivenProblem(sol::T; kwargs...) where T <: DiffEqBase.DESolution = DataDrivenProblem(sol; kwargs...)

function DataDrivenProblem(sol::T; use_interpolation = false, kwargs...) where T <: DiffEqBase.DESolution
    if sol.retcode != :Success
        throw(AssertionError("The solution is not successful. Abort."))
        return
    end

    X = Array(sol)
    t = sol.t
    p = sol.prob.p

    p = isa(p, DiffEqBase.NullParameters) ? eltype(X)[] : p

    if isdiscrete(sol.alg)

        return DiscreteDataDrivenProblem(
            X, t; p = p, kwargs...
        )

    else
        if use_interpolation
            DX = Array(sol(sol.t, Val{1}))
        else
            DX = similar(X)
            if DiffEqBase.isinplace(sol.prob.f)
                @views map(i->sol.prob.f(DX[:, i], X[:, i], p, t[i]), 1:size(X,2))
            else
                map(i->DX[:, i] .= sol.prob.f(X[:, i], p, t[i]), 1:size(X,2))
            end
        end

        return ContinuousDataDrivenProblem(
            X, t; DX = DX, p = p, kwargs...
        )
    end

end
