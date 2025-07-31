import Base: unique, unique!, ==, deleteat!

"""
$(TYPEDEF)

A basis over the states with parameters, independent variable, and possible exogenous controls.
It extends an `AbstractSystem` as defined in `ModelingToolkit.jl`. `f` can either be a Julia function which is able to use ModelingToolkit variables or
a vector of `eqs`.
It can be called with the typical SciML signature, meaning out of place with `f(u,p,t)`
or in place with `f(du, u, p, t)`. If control inputs are present, it is assumed that no control corresponds to
zero for all inputs. The corresponding function calls are `f(u,p,t,inputs)` and `f(du,u,p,t,inputs)` and need to
be specified fully.

The optional `implicits` declare implicit variables in the `Basis`, meaning variables representing the (measured) target of the system.
Right now, only supported with the use of `ImplicitOptimizer`s.

If `linear_independent` is set to `true`, a linear independent basis is created from all atom functions in `f`.

If `simplify_eqs` is set to `true`, `simplify` is called on `f`.

Additional keyword arguments include `name`, which can be used to name the basis, and
`observed` for defining observables.

# Fields

$(FIELDS)

# Example

```julia
using ModelingToolkit
using DataDrivenDiffEq

@parameters w[1:2] t
@variables u[1:2](t)

Ψ = Basis([u; sin.(w .* u)], u, parameters = p, iv = t)
```

## Note

The keyword argument `eval_expression` controls the function creation
behavior. `eval_expression=true` means that `eval` is used, so normal
world-age behavior applies (i.e. the functions cannot be called from
the function that generates them). If `eval_expression=false`,
then construction via GeneralizedGenerated.jl is utilized to allow for
same world-age evaluation. However, this can cause Julia to segfault
on sufficiently large basis functions. By default eval_expression=false.
"""
struct Basis{IMPL, CTRLS} <: AbstractBasis
    """The equations of the basis"""
    eqs::Vector{Equation}
    """Dependent (state) variables"""
    unknowns::Vector
    """Control variables"""
    ctrls::Vector
    """Parameters"""
    ps::Vector
    """Observed"""
    observed::Vector
    """Independent variable"""
    iv::Num
    """Implicit variables of the basis"""
    implicit::Vector
    """Internal function representation of the basis"""
    f::AbstractDataDrivenFunction{IMPL, CTRLS}
    """Name of the basis"""
    name::Symbol
    """Internal systems"""
    systems::Vector{AbstractBasis}

    function Basis(eqs, states, ctrls, ps, observed, iv, implicit, f, name, systems;
            checks::Bool = true)
        if checks
            # Currently do nothing here
            #check_variables(dvs, iv)
            #check_parameters(ps, iv)
            #check_equations(deqs, iv)
            #check_equations(equations(events), iv)
            #all_dimensionless([dvs; ps; iv]) || check_units(deqs)
        end

        imp_ = !isempty(implicit)
        ctrls_ = !isempty(ctrls)
        new{imp_, ctrls_}(eqs, states, ctrls, ps, observed, iv, implicit, f, name, systems)
    end
end

function __preprocess_basis(eqs, states, ctrls, ps, observed, iv, implicit, name, systems,
        simplify, linear_independent, eval_expression)
    # Check for iv
    iv === nothing && (iv = Symbolics.variable(:t))
    iv = value(iv)
    # Scalarize equations
    eqs = Symbolics.scalarize(eqs)

    lhs = isa(eqs, AbstractVector{Equation}) ?
          map(Base.Fix2(getfield, :lhs), eqs) :
          map(Base.Fix1(Symbolics.variable, :φ), 1:length(eqs))

    rhs = isa(eqs, AbstractVector{Equation}) ?
          map(Base.Fix2(getfield, :rhs), eqs) : eqs

    rhs = Num.(rhs)
    lhs = Num.(lhs)

    # Scalarize all variables
    states, controls,
    parameters,
    implicits,
    observed = value.(collect(states)),
    value.(collect(ctrls)),
    value.(collect(ps)),
    value.(collect(implicit)),
    value.(collect(observed))
    # Filter out zeros
    rhs = [eq for eq in rhs if ~isequal(Num(eq), zero(Num))]

    rhs = linear_independent ? create_linear_independent_eqs(rhs, false) : rhs
    unique!(rhs, simplify)

    f = DataDrivenFunction(rhs,
        implicits, states, parameters, iv,
        controls, eval_expression)

    eqs = reduce(vcat, map(Symbolics.Equation, lhs, rhs); init = Equation[])
    eqs = isa(eqs, AbstractVector) ? collect(eqs) : [collect(eqs)]
    return eqs, states, controls, parameters, observed, iv, implicits, f, name,
    systems
end

## Constructors

function Basis(eqs::AbstractVector, states::AbstractVector;
        parameters::AbstractVector = [], iv = nothing,
        controls::AbstractVector = [], implicits = [],
        observed::AbstractVector = [],
        name = gensym(:Basis),
        simplify = false, linear_independent = false,
        eval_expression = false,
        kwargs...)
    return Basis(__preprocess_basis(eqs, states, controls, parameters, observed, iv,
        implicits, name, AbstractBasis[], simplify,
        linear_independent, eval_expression)...)
end

function Basis(f::Function, states::AbstractVector; parameters::AbstractVector = [],
        controls::AbstractVector = [], implicits::AbstractVector = [],
        iv = nothing, kwargs...)
    isnothing(iv) && (iv = Num(Variable(:t)))

    try
        if isempty(controls) && isempty(implicits)
            eqs = f(states, parameters, iv)
        elseif isempty(controls)
            eqs = f(implicits, states, parameters, iv)
        elseif isempty(implicits)
            eqs = f(states, parameters, iv, controls)
        else
            eqs = f(implicits, states, parameters, iv, controls)
        end
        return Basis(eqs, states, parameters = parameters, iv = iv, controls = controls,
            implicits = implicits; kwargs...)
    catch e
        rethrow(e)
    end
end

## Printing

@inline function Base.print(io::IO, x::AbstractBasis)
    state = unknowns(x)
    ps = parameters(x)
    Base.printstyled(io, "Model $(nameof(x)) with $(length(x)) equations\n"; bold = true)
    print(io, "States :")
    if length(state) < 5
        for xi in state
            print(io, " ", xi)
        end
    else
        print(io, " ", length(state))
    end

    if !isempty(ps)
        print(io, "\nParameters :")
        if length(ps) < 5
            for p_ in ps
                print(io, " ", p_)
            end
        else
            print(io, " ", length(ps))
        end
    end

    println(io, "\nIndependent variable: $(get_iv(x))")
    println(io, "Equations")
    for (i, eq) in enumerate(equations(x))
        if i < 5 || i == length(x)
            println(io, "$(eq.lhs) = $(eq.rhs)")
        elseif i == 5
            println(io, "...")
        else
            continue
        end
    end
end

@inline function Base.print(io::IO, x::AbstractBasis, fullview::Bool)
    !fullview && return print(io, x)

    state = unknowns(x)
    ps = parameters(x)
    Base.printstyled(io, "Model $(nameof(x)) with $(length(x)) equations\n"; bold = true)
    print(io, "States :")
    if length(state) < 5
        for xi in state
            print(io, " ", xi)
        end
    else
        print(io, " ", length(state))
    end

    if !isempty(ps)
        print(io, "\nParameters :")
        if length(ps) < 5
            for p_ in ps
                print(io, " ", p_)
            end
        else
            print(io, " ", length(ps))
        end
    end

    println(io, "\nIndependent variable: $(get_iv(x))")
    println(io, "Equations")
    for (i, eq) in enumerate(equations(x))
        println(io, "$i : $(eq.lhs) = $(eq.rhs)")
    end
end

## Getters

"""
    $(SIGNATURES)

    Returns the internal function representing the dynamics of the `Basis`. This can be called either inplace or out-of-place
    with the typical SciML signature `f(u,p,t)` or `f(du,u,p,t)`. If control variables are defined, the function can also be called
    by `f(u,p,t,control)` or `f(du,u,p,t,control)` and assumes `control .= 0` if no control is given.
"""
function dynamics(b::AbstractBasis)
    return get_f(b)
end

"""
$(SIGNATURES)

Return the implicit variables of the basis.
"""
function implicit_variables(b::AbstractBasis)
    return getfield(b, :implicit)
end

function states(b::AbstractBasis)
    return getfield(b, :unknowns)
end

function controls(b::AbstractBasis)
    ctrls = getfield(b, :ctrls)
    systems = getfield(b, :systems)
    isempty(systems) && return ctrls

    ctrls = copy(ctrls)
    for sys in systems
        append!(ctrls, unknowns(sys, controls(sys)))
    end
    return ctrls
end

# For internal use
is_implicit(b::Basis{X, <:Any}) where {X} = X
is_controlled(b::Basis{<:Any, X}) where {X} = X

## Callable
get_f(b::AbstractBasis) = getfield(b, :f)

#(b::Basis)(args...) = get_f(b)(args...)
# OOP

# Without controls or implicits
function (b::Basis{false, false})(u::AbstractVector, p::P = parameters(b),
        t::Number = get_iv(b)) where {
        P <:
        Union{AbstractArray, Tuple}}
    f = get_f(b)
    f(u, p, t)
end

# Without implicits, with controls
function (b::Basis{false, true})(u::AbstractVector,
        p::P = parameters(b), t::Number = get_iv(b),
        c::AbstractVector = controls(b)) where {
        P <: Union{
        AbstractArray,
        Tuple}}
    f = get_f(b)
    f(u, p, t, c)
end

# With implicit, without controls
function (b::Basis{true, false})(du::AbstractVector, u::AbstractVector,
        p::P = parameters(b),
        t::Number = get_iv(b)) where {
        P <:
        Union{AbstractArray, Tuple}}
    f = get_f(b)
    f(du, u, p, t)
end

# With implicit and controls
function (b::Basis{true, true})(du::AbstractVector, u::AbstractVector,
        p::P = parameters(b), t::Number = get_iv(b),
        c::AbstractVector = controls(b)) where {
        P <:
        Union{AbstractArray,
        Tuple}}
    f = get_f(b)
    f(du, u, p, t, c)
end

# Array
function (b::Basis{false, false})(u::AbstractMatrix, p::P,
        t::AbstractVector) where {P <:
                                  Union{AbstractArray, Tuple}}
    f = get_f(b)
    f(u, p, t)
end

function (b::Basis{true, false})(du::AbstractMatrix, u::AbstractMatrix, p::P,
        t::AbstractVector) where {P <: Union{AbstractArray, Tuple}}
    f = get_f(b)
    f(du, u, p, t)
end

function (b::Basis{false, true})(u::AbstractMatrix, p::P, t::AbstractVector,
        c::AbstractMatrix) where {P <: Union{AbstractArray, Tuple}}
    f = get_f(b)
    f(u, p, t, c)
end

function (b::Basis{true, true})(du::AbstractMatrix, u::AbstractMatrix, p::P,
        t::AbstractVector,
        c::AbstractMatrix) where {P <: Union{AbstractArray, Tuple}}
    f = get_f(b)
    f(du, u, p, t, c)
end

function (b::Basis{false, false})(res::AbstractMatrix, u::AbstractMatrix, p::P,
        t::AbstractVector) where {P <:
                                  Union{AbstractArray, Tuple}}
    f = get_f(b)
    f(res, u, p, t)
end

function (b::Basis{true, false})(
        res::AbstractMatrix, du::AbstractMatrix, u::AbstractMatrix,
        p::P,
        t::AbstractVector) where {P <: Union{AbstractArray, Tuple}}
    f = get_f(b)
    f(res, du, u, p, t)
end

function (b::Basis{false, true})(res::AbstractMatrix, u::AbstractMatrix, p::P,
        t::AbstractVector,
        c::AbstractMatrix) where {P <: Union{AbstractArray, Tuple}}
    f = get_f(b)
    f(res, u, p, t, c)
end

function (b::Basis{true, true})(res::AbstractMatrix, du::AbstractMatrix, u::AbstractMatrix,
        p::P, t::AbstractVector,
        c::AbstractMatrix) where {P <: Union{AbstractArray, Tuple}}
    f = get_f(b)
    f(res, du, u, p, t, c)
end

## Information and Iteration

Base.length(x::B) where {B <: AbstractBasis} = length(equations(x))
Base.size(x::B) where {B <: AbstractBasis} = size(equations(x))

Base.getindex(x::B, idx) where {B <: AbstractBasis} = getindex(equations(x), idx)
Base.firstindex(x::B) where {B <: AbstractBasis} = firstindex(equations(x))
Base.lastindex(x::B) where {B <: AbstractBasis} = lastindex(equations(x))
Base.iterate(x::B) where {B <: AbstractBasis} = iterate(equations(x))
Base.iterate(x::B, id) where {B <: AbstractBasis} = iterate(equations(x), id)

## Internal update
function __update!(b::AbstractBasis, eval_expression = false)
    ff = DataDrivenFunction([bi.rhs for bi in collect(equations(b))],
        implicit_variables(b), unknowns(b), parameters(b), [get_iv(b)],
        controls(b), eval_expression)
    @set! b.f = ff
    return
end

function Base.setindex!(x::AbstractBasis, idx, val, eval_expression = false)
    setindex!(equations(x), idx, val)
    __update!(x, eval_expression)
    return
end

## Derivatives

"""
$(SIGNATURES)

Returns a function representing the Jacobian matrix / gradient of the [`Basis`](@ref) with respect to the
states as a function with the common signature `f(u,p,t)` for out of place and `f(du, u, p, t)` for in place computation.
If control variables are defined, the function can also be called by `f(u,p,t,control)` or `f(du,u,p,t,control)` and assumes `control .= 0` if no control is given.

If the Jacobian with respect to other variables is needed, it can be passed via a second argument.
"""
function jacobian(x::Basis, eval_expression::Bool = false)
    jacobian(
        x, unknowns(x), eval_expression)
end

function jacobian(x::Basis, s, eval_expression::Bool = false)
    j = Symbolics.jacobian([xi.rhs for xi in equations(x)], s)

    return DataDrivenFunction(j,
        implicit_variables(x), unknowns(x), parameters(x), [get_iv(x)],
        controls(x), eval_expression)
end

## Utilities
function Base.deleteat!(b::Symbolics.Arr{T, N}, idxs) where {T, N}
    deleteat!(Symbolics.unwrap(b), idxs)
end

## Interfacing && merging

function Base.unique!(b::AbstractVector{Num}, simplify_eqs = false)
    idx = zeros(Bool, length(b))
    for i in 1:length(b), j in (i + 1):length(b)

        i == j && continue
        idx[i] && continue
        idx[i] = isequal(b[i], b[j])
    end
    deleteat!(b, idx)
    simplify_eqs && map(ModelingToolkit.simplify, b)
    return
end

function Base.unique!(b::Basis, simplify_eqs = false; eval_expression = false)
    idx = zeros(Bool, length(b))
    eqs_ = equations(b)
    n_eqs = length(eqs_)
    for i in 1:n_eqs, j in (i + 1):n_eqs

        i == j && continue
        idx[i] && continue
        idx[i] = isequal(eqs_[i].rhs, eqs_[j].rhs)
    end
    deleteat!(equations(b), idx)
    simplify_eqs && map(ModelingToolkit.simplify, equations(b))
    __update!(b, eval_expression)
end

function Base.deleteat!(b::Basis, inds; eval_expression = false)
    deleteat!(equations(b), inds)
    __update!(b, eval_expression)
    return
end

function Base.push!(b::Basis, eqs::AbstractArray, simplify_eqs = true;
        eval_expression = false)
    @inbounds for eq in eqs
        push!(b, eq, false)
    end
    unique!(b, simplify_eqs, eval_expression = eval_expression)
    return
end

function Base.push!(b::Basis, eq, simplify_eqs = true; eval_expression = false)
    push!(equations(b), variable(:φ, length(b) + 1) ~ eq)
    unique!(b, simplify_eqs, eval_expression = eval_expression)
    return
end

function Base.push!(b::Basis, eq::Equation, simplify_eqs = true; eval_expression = false)
    push!(equations(b), eq)
    unique!(b, simplify_eqs, eval_expression = eval_expression)
    return
end

function Base.merge(x::Basis, y::Basis; eval_expression = false)
    x_ = deepcopy(x)
    merge!(x_, y, eval_expression = eval_expression)
    return x_
end

function Base.merge!(x::Basis, y::Basis; eval_expression = false)
    push!(x, equations(y))
    @set! x.unknowns = unique(vcat(unknowns(x), unknowns(y)))
    @set! x.ps = unique(vcat(parameters(x), parameters(y)))
    @set! x.ctrls = unique(vcat(controls(x), controls(y)))
    @set! x.observed = unique(vcat(get_observed(x), get_observed(y)))
    __update!(x, eval_expression)
    return
end

## Additional functionalities

function Base.isequal(x::Basis, y::Basis)
    length(x) == length(y) || return false
    yrhs = [yi.rhs for yi in equations(y)]
    xrhs = [xi.rhs for xi in equations(x)]
    isequal(yrhs, xrhs)
end

"""
$(SIGNATURES)

Return the default values for the given [`Basis`](@ref).
If no default value is stored, returns `zero(T)` where `T` is the `symtype` of the parameter.

## Note

This extends `getmetadata` in a way that all parameters have a numeric value.
"""
function get_parameter_values(x::Basis)
    map(parameters(x)) do p
        if hasmetadata(p, Symbolics.VariableDefaultValue)
            return Symbolics.getdefaultval(p)
        else
            return zero(Symbolics.symtype(p))
        end
    end
end

"""
$(SIGNATURES)

Return the default values as a vector of pairs for the given [`Basis`](@ref).
If no default value is stored, returns `zero(T)` where `T` is the `symtype` of the parameter.

## Note

This extends `getmetadata` in a way that all parameters have a numeric value.
"""
function get_parameter_map(x::Basis)
    map(parameters(x)) do p
        if hasmetadata(p, Symbolics.VariableDefaultValue)
            return p => Symbolics.getdefaultval(p)
        else
            return p => zero(Symbolics.symtype(p))
        end
    end
end
