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
Right now only supported with the use of `ImplicitOptimizer`s.


If `linear_independent` is set to `true`, a linear independent basis is created from all atom functions in `f`.

If `simplify_eqs` is set to `true`, `simplify` is called on `f`.

Additional keyworded arguments include `name`, which can be used to name the basis, and
`observed` for defining observables.


# Fields
$(FIELDS)

# Example

```julia
using ModelingToolkit
using DataDrivenDiffEq

@parameters w[1:2] t
@variables u[1:2](t)

Ψ = Basis([u; sin.(w.*u)], u, parameters = p, iv = t)
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
mutable struct Basis{I} <: AbstractBasis{I}
    """The equations of the basis"""
    eqs::Vector{Equation}
    """Dependent (state) variables"""
    states::Vector
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
    f::Function
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
        new{imp_}(eqs, states, ctrls, ps, observed, iv, implicit, f, name, systems)
    end
end

function __preprocess_basis(eqs, states, ctrls, ps, observed, iv, implicit, name, systems,
                            simplify, linear_independent, eval_expression)
    # Check for iv
    iv === nothing && (iv = Symbolics.variable(:t))
    iv = value(iv)
    # Scalarize equations
    eqs = scalarize(eqs)
    lhs = isa(eqs, AbstractVector{Symbolics.Equation}) ?
          map(Base.Fix2(getfield, :lhs), eqs) :
          map(Base.Fix1.(Symbolics.variable, :φ), 1:length(eqs))
    rhs = isa(eqs, AbstractVector{Symbolics.Equation}) ?
          map(Base.Fix2(getfield, :lhs), eqs) : eqs

    # Scalarize all variables
    states, controls, parameters, implicits, observed = value.(scalarize(states)),
                                                        value.(scalarize(ctrls)),
                                                        value.(scalarize(ps)),
                                                        value.(scalarize(implicit)),
                                                        value.(scalarize(observed))
    # Filter out zeros
    rhs = [eq for eq in rhs if ~isequal(Num(eq), zero(Num))]

    rhs = linear_independent ? create_linear_independent_eqs(rhs, false) : rhs
    unique!(rhs, simplify)

    f = _build_ddd_function(rhs, [states; implicits], parameters, iv, controls,
                            eval_expression)

    eqs = reduce(vcat, map(Symbolics.Equation, lhs, rhs))
    return collect(eqs), states, ctrls, ps, observed, iv, implicit, f, name, systems
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
    #return __preprocess_basis(eqs, states, controls, parameters, observed, iv, implicits, name, AbstractBasis[], simplify, linear_independent, eval_expression)

    return Basis(__preprocess_basis(eqs, states, controls, parameters, observed, iv,
                                    implicits, name, AbstractBasis[], simplify,
                                    linear_independent, eval_expression)...)
end

function Basis(f::Function, states::AbstractVector; parameters::AbstractVector = [],
               controls::AbstractVector = [], implicits::AbstractVector = [],
               iv = nothing, kwargs...)
    isnothing(iv) && (iv = Num(Variable(:t)))

    try
        eqs = isempty(controls) ? f([states; implicits], parameters, iv) :
              f([states; implicits], parameters, iv, controls)
        return Basis(eqs, states, parameters = parameters, iv = iv, controls = controls,
                     implicits = implicits; kwargs...)
    catch e
        rethrow(e)
    end
end

## Printing

@inline function Base.print(io::IO, x::AbstractBasis)
    state = states(x)
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

    state = states(x)
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
        println(io, "$(eq.lhs) = $(eq.rhs)")
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

# For internal use
is_implicit(b::AbstractBasis{X}) where {X} = X

## Callable
get_f(b::AbstractBasis) = getfield(b, :f)

# Fallback
(b::AbstractBasis)(args...) = get_f(b)(args...)

# OOP
function (b::AbstractBasis)(x::AbstractVector{T} where {T},
                            p::AbstractVector{T} where {T} = parameters(b),
                            t::T where {T <: Number} = get_iv(b))
    return get_f(b)(x, p, t)
end

function (b::AbstractBasis)(x::AbstractVector{T} where {T}, p::AbstractVector{T} where {T},
                            t::T where {T <: Number}, u::AbstractVector{T} where {T})
    return get_f(b)(x, p, t, u)
end

function (b::AbstractBasis)(x::AbstractMatrix{T} where {T})
    t = get_iv(b)
    return get_f(b)(x, parameters(b), [t for i in 1:size(x, 2)])
end

function (b::AbstractBasis)(x::AbstractMatrix{T} where {T}, p::AbstractVector{T} where {T})
    t = get_iv(b)
    return get_f(b)(x, p, [t for i in 1:size(x, 2)])
end

function (b::AbstractBasis)(x::AbstractMatrix{T} where {T}, p::AbstractVector{T} where {T},
                            t::AbstractVector{T} where {T <: Number})
    return get_f(b)(x, p, t)
end

function (b::AbstractBasis)(x::AbstractMatrix{T} where {T}, p::AbstractVector{T} where {T},
                            t::AbstractVector{T} where {T <: Number},
                            u::AbstractMatrix{T} where {T})
    return get_f(b)(x, p, t, u)
end

# IIP
function (b::AbstractBasis)(y::AbstractMatrix{T} where {T}, x::AbstractMatrix{T} where {T})
    t = get_iv(b)
    return get_f(b)(y, x, parameters(b), [t for i in 1:size(x, 2)])
end

function (b::AbstractBasis)(y::AbstractMatrix{T} where {T}, x::AbstractMatrix{T} where {T},
                            p::AbstractVector{T} where {T})
    t = get_iv(b)
    return get_f(b)(y, x, p, [t for i in 1:size(x, 2)])
end

function (b::AbstractBasis)(y::AbstractMatrix{T} where {T}, x::AbstractMatrix{T} where {T},
                            p::AbstractVector{T} where {T},
                            t::AbstractVector{T} where {T <: Number})
    return get_f(b)(y, x, p, t)
end

function (b::AbstractBasis)(y::AbstractMatrix{T} where {T}, x::AbstractMatrix{T} where {T},
                            p::AbstractVector{T} where {T},
                            t::AbstractVector{T} where {T <: Number},
                            u::AbstractMatrix{T} where {T})
    return get_f(b)(y, x, p, t, u)
end

## Information and Iteration

Base.length(x::AbstractBasis) = length(equations(x))
Base.size(x::AbstractBasis) = size(equations(x))

Base.getindex(x::AbstractBasis, idx) = getindex(equations(x), idx)
Base.firstindex(x::AbstractBasis) = firstindex(equations(x))
Base.lastindex(x::AbstractBasis) = lastindex(equations(x))
Base.iterate(x::AbstractBasis) = iterate(equations(x))
Base.iterate(x::AbstractBasis, id) = iterate(equations(x), id)

## Internal update
function __update!(b::AbstractBasis, eval_expression = false)
    ff = _build_ddd_function([bi.rhs for bi in collect(equations(b))],
                             states(b), parameters(b), [get_iv(b)],
                             controls(b), eval_expression)
    setfield!(b, :f, ff)
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

Returns a function representing the jacobian matrix / gradient of the [`Basis`](@ref) with respect to the
states as a function with the common signature `f(u,p,t)` for out of place and `f(du, u, p, t)` for in place computation.
If control variables are defined, the function can also be called by `f(u,p,t,control)` or `f(du,u,p,t,control)` and assumes `control .= 0` if no control is given.

If the jacobian with respect to other variables is needed, it can be passed via a second argument.
"""
jacobian(x::Basis, eval_expression::Bool = false) = jacobian(x, states(x), eval_expression)

function jacobian(x::Basis, s, eval_expression::Bool = false)
    j = Symbolics.jacobian([xi.rhs for xi in equations(x)], s)

    return _build_ddd_function(j,
                               states(x), parameters(x), get_iv(x),
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

"""
$(SIGNATURES)

Removes duplicate equations from the [`Basis`](@ref).
"""
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

"""
$(SIGNATURES)

Delete the entries specified by `inds` and update the [`Basis`](@ref) accordingly.
"""
function Base.deleteat!(b::Basis, inds; eval_expression = false)
    deleteat!(equations(b), inds)
    __update!(b, eval_expression)
    return
end

"""
$(SIGNATURES)

Append the provided elements to the [`Basis`](@ref) as an [`Symbolics.Equation`](@ref).
"""
function Base.push!(b::Basis, eqs::AbstractArray, simplify_eqs = true;
                    eval_expression = false)
    @inbounds for eq in eqs
        push!(b, eq, false)
    end
    unique!(b, simplify_eqs, eval_expression = eval_expression)
    return
end

"""
$(SIGNATURES)

Append the provided element to the [`Basis`](@ref) as an [`Symbolics.Equation`](@ref).
"""
function Base.push!(b::Basis, eq, simplify_eqs = true; eval_expression = false)
    push!(equations(b), variable(:φ, length(b) + 1) ~ eq)
    unique!(b, simplify_eqs, eval_expression = eval_expression)
    return
end

"""
$(SIGNATURES)

Append the provided [`Symbolics.Equation`](@ref) to the [`Basis`](@ref).
"""
function Base.push!(b::Basis, eq::Equation, simplify_eqs = true; eval_expression = false)
    push!(equations(b), eq)
    unique!(b, simplify_eqs, eval_expression = eval_expression)
    return
end

"""
$(SIGNATURES)

Merges the provided [`Basis`](@ref) and returns a new [`Basis`](@ref).
"""
function Base.merge(x::Basis, y::Basis; eval_expression = false)
    x_ = deepcopy(x)
    merge!(x_, y, eval_expression = eval_expression)
    return x_
end

"""
$(SIGNATURES)

Merges the provided [`Basis`](@ref) inplace.
"""
function Base.merge!(x::Basis, y::Basis; eval_expression = false)
    push!(x, equations(y))
    setfield!(x, :states, unique(vcat(states(x), states(y))))
    setfield!(x, :ps, unique(vcat(parameters(x), parameters(y))))
    setfield!(x, :ctrls, unique(vcat(controls(x), controls(y))))
    setfield!(x, :observed, unique(vcat(get_observed(x), get_observed(y))))
    __update!(x, eval_expression)
    return
end

## Additional functionalities
"""
$(SIGNATURES)

Check for equality of the provided [`Basis`](@ref).
"""
function Base.isequal(x::Basis, y::Basis)
    length(x) == length(y) || return false
    yrhs = [yi.rhs for yi in equations(y)]
    xrhs = [xi.rhs for xi in equations(x)]
    isequal(yrhs, xrhs)
end
