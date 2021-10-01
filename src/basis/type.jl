import Base: unique, unique!, ==, deletat!


"""
$(TYPEDEF)

A basis over the states with parameters, independent variable, and possible exogenous controls.
It extends an `AbstractSystem` as defined in `ModelingToolkit.jl`. `f` can either be a Julia function which is able to use ModelingToolkit variables or
a vector of `eqs`.
It can be called with the typical SciML signature, meaning out of place with `f(u,p,t)`
or in place with `f(du, u, p, t)`. If control inputs are present, it is assumed that no control corresponds to
zero for all inputs. The corresponding function calls are `f(u,p,t,inputs)` and `f(du,u,p,t,inputs)` and need to
be specified fully.

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
mutable struct Basis <: AbstractBasis
    """The equations of the basis"""
    eqs::Vector{Equation}
    """Dependent (state) variables"""
    states::Vector
    """Control variables"""
    controls::Vector
    """Parameters"""
    ps::Vector
    """Observed"""
    observed::Vector
    """Independent variable"""
    iv::Num
    """Internal function representation of the basis"""
    f::Function
    """Name of the basis"""
    name::Symbol
    """Internal systems"""
    systems::Vector{Basis}
end

## Constructors

function Basis(eqs::AbstractVector, states::AbstractVector;
    parameters::AbstractVector = [], iv = nothing,
    controls::AbstractVector = [], observed::AbstractVector = [],
    name = gensym(:Basis),
    simplify = false, linear_independent = false,
    eval_expression = false,
    kwargs...)
    iv === nothing && (iv = Symbolics.variable(:t))
    iv = value(iv)
    eqs = scalarize(eqs)
    states, controls, parameters, observed = value.(scalarize(states)), value.(scalarize(controls)), value.(scalarize(parameters)), value.(scalarize(observed))

    eqs = [eq for eq in eqs if ~isequal(Num(eq),zero(Num))]

    if linear_independent
        eqs_ = create_linear_independent_eqs(eqs, simplify)
    else
        eqs_ = simplify ? ModelingToolkit.simplify.(eqs) : eqs
    end

    unique!(eqs, !simplify)

    f = _build_ddd_function(eqs, states, parameters, iv, controls, eval_expression)

    eqs = [Symbolics.variable(:φ,i) ~ eq for (i,eq) ∈ enumerate(eqs_)]

    return Basis(collect(eqs), states, controls, parameters, observed, iv, f, name, Basis[])
end


function Basis(eqs::AbstractVector{Equation}, states::AbstractVector;
    parameters::AbstractVector = [], iv = nothing,
    controls::AbstractVector = [], observed::AbstractVector = [],
    name = gensym(:Basis),
    simplify = false, linear_independent = false,
    eval_expression = false,
    kwargs...)

    lhs = [x.lhs for x in eqs]
    # We filter out 0s
    eqs_ = [Num(x.rhs) for x in eqs if ~isequal(Num(x),zero(Num))]
    if linear_independent
        eqs_ = create_linear_independent_eqs(eqs_, simplify)
    else
        eqs_ = simplify ? ModelingToolkit.simplify.(eqs_) : eqs_
    end

    isnothing(iv) && (iv = Num(variable(:t)))
    unique!(eqs_, !simplify)

    f = _build_ddd_function(eqs_, states, parameters, iv, controls, eval_expression)

    eqs = [lhs[i] ~ eq for (i,eq) ∈ enumerate(eqs_)]

    return Basis(collect(eqs), value.(states), value.(controls), value.(parameters), value.(observed), value(iv), f, name, Basis[])

end


function Basis(f::Function, states::AbstractVector; parameters::AbstractVector = [], controls::AbstractVector = [],
     iv = nothing, kwargs...)

    isnothing(iv) && (iv = Num(Variable(:t)))

    try
        eqs = isempty(controls) ? f(states, parameters, iv) : f(states, parameters, iv, controls)
        return Basis(eqs, states, parameters = parameters, iv = iv, controls = controls; kwargs...)
    catch e
        rethrow(e)
    end
end

function Basis(n::Int)
    @parameters t
    @variables u[1:n](t)
    return Basis(u, u, iv = t)
end

## Printing

@inline function Base.print(io::IO, x::AbstractBasis)
    state = states(x)
    ps = parameters(x)
    Base.printstyled(io, "Model $(nameof(x)) with $(length(x)) equations\n"; bold=true)
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

    println(io, "\nIndependent variable: $(independent_variable(x))")
    println(io, "Equations")
    for (i,eq) ∈ enumerate(equations(x))
        if i < 5 || i == length(x)
        println(io, "$(eq.lhs) = $(eq.rhs)")
        elseif i == 5
            println(io, "...")
        else
            continue
        end
    end
end

@inline function Base.println(io::IO, x::AbstractBasis, fullview::DataType = Val{false})
    fullview == Val{false} && return print(io, x)

    state = states(x)
    ps = parameters(x)
    Base.printstyled(io, "Model $(nameof(x)) with $(length(x)) equations\n"; bold=true)
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

    println(io, "\nIndependent variable: $(independent_variable(x))")
    println(io, "Equations")
    for (i,eq) ∈ enumerate(equations(x))
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
Returns the control variables of the `Basis`.

$(SIGNATURES)
"""
ModelingToolkit.controls(b::AbstractBasis) = b.controls

## Callable

# Fallback
(b::AbstractBasis)(args...) = b.f(args...)

# OOP
function (b::AbstractBasis)(x::AbstractVector{T} where T, p::AbstractVector{T} where T = parameters(b),
    t::T where T <: Number = independent_variable(b))
    return b.f(x,p,t)
end

function (b::AbstractBasis)(x::AbstractVector{T} where T, p::AbstractVector{T} where T,
    t::T where T <: Number , u::AbstractVector{T} where T)
    return b.f(x,p,t, u)
end

function (b::AbstractBasis)(x::AbstractMatrix{T} where T)
    t = independent_variable(b)
    return b.f(x,parameters(b),[t for i in 1:size(x,2)])
end

function (b::AbstractBasis)(x::AbstractMatrix{T} where T, p::AbstractVector{T} where T,)
    t = independent_variable(b)
    return b.f(x,p,[t for i in 1:size(x,2)])
end

function (b::AbstractBasis)(x::AbstractMatrix{T} where T, p::AbstractVector{T} where T,
    t::AbstractVector{T} where T <: Number)
    return b.f(x,p,t)
end


function (b::AbstractBasis)(x::AbstractMatrix{T} where T, p::AbstractVector{T} where T,
    t::AbstractVector{T} where T <: Number, u::AbstractMatrix{T} where T)
    return b.f(x,p,t, u)
end

# IIP
function (b::AbstractBasis)(y::AbstractMatrix{T} where T, x::AbstractMatrix{T} where T)
    t = independent_variable(b)
    return b.f(y,x,parameters(b),[t for i in 1:size(x,2)])
end

function (b::AbstractBasis)(y::AbstractMatrix{T} where T, x::AbstractMatrix{T} where T, p::AbstractVector{T} where T,)
    t = independent_variable(b)
    return b.f(y,x,p,[t for i in 1:size(x,2)])
end

function (b::AbstractBasis)(y::AbstractMatrix{T} where T, x::AbstractMatrix{T} where T, p::AbstractVector{T} where T,
    t::AbstractVector{T} where T <: Number, )
    return b.f(y,x,p,t)
end

function (b::AbstractBasis)(y::AbstractMatrix{T} where T, x::AbstractMatrix{T} where T, p::AbstractVector{T} where T,
    t::AbstractVector{T} where T <: Number, u::AbstractMatrix{T} where T)
    return b.f(y,x,p,t,u)
end

## Information and Iteration

Base.length(x::AbstractBasis) = length(x.eqs)
Base.size(x::AbstractBasis) = size(x.eqs)

Base.getindex(x::AbstractBasis, idx) = getindex(equations(x), idx)
Base.firstindex(x::AbstractBasis) = firstindex(equations(x))
Base.lastindex(x::AbstractBasis) = lastindex(equations(x))
Base.iterate(x::AbstractBasis) = iterate(equations(x))
Base.iterate(x::AbstractBasis, id) = iterate(equations(x), id)

## Internal update

function update!(b::AbstractBasis, eval_expression = false)

    ff = _build_ddd_function([bi.rhs for bi in collect(equations(b))],
        states(b), parameters(b), [independent_variable(b)],
        controls(b), eval_expression)
    Core.setfield!(b, :f, ff)

    return
end


function Base.setindex!(x::AbstractBasis, idx, val, eval_expression = false)
    setindex!(equations(x), idx, val)
    update!(x, eval_expression)
    return
end

## Derivatives


"""
    $(SIGNATURES)

    Returns a function representing the jacobian matrix / gradient of the `Basis` with respect to the
    states as a function with the common signature `f(u,p,t)` for out of place and `f(du, u, p, t)` for in place computation.
    If control variables are defined, the function can also be called by `f(u,p,t,control)` or `f(du,u,p,t,control)` and assumes `control .= 0` if no control is given.

    If the jacobian with respect to other variables is needed, it can be passed via a second argument.
"""
function jacobian(x::Basis, eval_expression::Bool = false)

    j = Symbolics.jacobian([xi.rhs for xi in equations(x)], states(x))

    jac  = _build_ddd_function(expand_derivatives.(j),
        states(x), parameters(x), independent_variable(x),
        controls(x), eval_expression)

    return jac
end


function jacobian(x::Basis, s, eval_expression::Bool = false)

    j = Symbolics.jacobian([xi.rhs for xi in equations(x)], s)

    jac  = _build_ddd_function(expand_derivatives.(j),
        states(x), parameters(x), independent_variable(x),
        controls(x), eval_expression)

    return jac
end

## Utilities
function Base.deleteat!(b::Symbolics.Arr{T,N}, idxs) where {T,N}
    #@show Symbolics.unwrap(b) #b = collect(b)
    deleteat!(Symbolics.unwrap(b), idxs)
end

function unique(b::AbstractArray{Num}, simplify_eqs::Bool)
    b = simplify_eqs ? simplify.(b) : b
    returns = ones(Bool, size(b)...)
    N = maximum(eachindex(b))
    for i ∈ eachindex(b)
        returns[i] = !any([isequal(b[i], b[j]) for j in i+1:N])
    end
    return Num.(b[returns])
end


function Base.unique!(b::AbstractArray{Num}, simplify_eqs = false)
    bs = simplify_eqs ? simplify.(b) : b
    removes = zeros(Bool, size(bs)...)
    N = maximum(eachindex(bs))
    for i ∈ eachindex(bs)
        removes[i] = any([isequal(bs[i], bs[j]) for j in i+1:N])
    end
    deleteat!(b, removes)
end

function unique(b::AbstractArray{Equation}, simplify_eqs::Bool)
    b = simplify_eqs ? simplify.(b) : b
    returns = ones(Bool, size(b)...)
    N = maximum(eachindex(b))
    for i ∈ eachindex(b)
        returns[i] = !any([isequal(b[i].rhs, b[j].rhs) for j in i+1:N])
    end
    return b[returns]
end

function Base.unique!(b::AbstractArray{Equation}, simplify_eqs::Bool)
    bs = [bi.rhs for bi in b]
    bs = simplify_eqs ? simplify.(bs) : bs
    removes = zeros(Bool, size(bs)...)
    N = maximum(eachindex(bs))
    for i ∈ eachindex(bs)
        removes[i] = any([isequal(bs[i], bs[j]) for j in i+1:N])
    end
    deleteat!(b, removes)
end

## Interfacing && merging

function Base.unique!(b::Basis, simplify_eqs = false; eval_expression = false)
    unique!(b.eqs, simplify_eqs)
    update!(b, eval_expression)
end

function Base.unique(b::Basis; kwargs...)
    eqs = unique(equations(b))
    return Basis(eqs, states(b), parameters = parameters(b), iv = independent_variable(b), kwargs...)
end

"""
    $(SIGNATURES)

    Delete the entries specified by `inds` and update the `Basis` accordingly.
"""
function Base.deleteat!(b::Basis, inds; eval_expression = false)
    deleteat!(b.eqs, inds)
    update!(b, eval_expression)
    return
end

"""
    $(SIGNATURES)

    Push the equations(s) in `eq` into the basis and update all internal fields accordingly.
    `eq` can either be a single equation or an array. If `simplify_eq` is true, the equation will be simplified.
"""
function Base.push!(b::Basis, eqs::AbstractArray, simplify_eqs = true; eval_expression = false)
    @inbounds for eq ∈ eqs
        push!(b, eq, false)
    end
    unique!(b, simplify_eqs, eval_expression = eval_expression)
    return
end

function Base.push!(b::Basis, eq, simplify_eqs = true; eval_expression = false)
    push!(equations(b), variable(:φ, length(b.eqs)+1)~eq)
    unique!(b, simplify_eqs, eval_expression = eval_expression)
    return
end

function Base.push!(b::Basis, eq::Equation, simplify_eqs = true; eval_expression = false)
    push!(equations(b), eq)
    unique!(b, simplify_eqs, eval_expression = eval_expression)
    return
end


"""
    $(SIGNATURES)

    Return a new `Basis`, which is defined via the union of `x` and `y` .
"""
function Base.merge(x::Basis, y::Basis; eval_expression = false)
    b =  unique(vcat([xi.rhs  for xi ∈ equations(x)], [xi.rhs  for xi ∈ equations(y)]))
    vs = unique(vcat(states(x), states(y)))
    ps = unique(vcat(parameters(x), parameters(y)))

    c = unique(vcat(controls(x), controls(y)))
    observed = unique(vcat(get_observed(x), get_observed(y)))

    return Basis(Num.(b), vs, parameters = ps, controls = c, observed = observed, eval_expression = eval_expression)
end

"""
    $(SIGNATURES)

    Updates `x` to include the union of both `x` and `y`.
"""
function Base.merge!(x::Basis, y::Basis; eval_expression = false)
    push!(x, map(x->x.rhs, equations(y)))
    Core.setfield!(x, :states, unique(vcat(states(x),states(y))))
    Core.setfield!(x, :ps, unique(vcat(parameters(x), parameters(y))))
    Core.setfield!(x, :controls, unique(vcat(controls(x), controls(y))))
    Core.setfield!(x, :observed, unique(vcat(get_observed(x), get_observed(y))))
    update!(x, eval_expression)
    return
end

## Additional functionalities

function (==)(x::Basis, y::Basis)
    length(x) == length(y) || return false
    n = zeros(Bool, length(x))
    yrhs = [yi.rhs for yi in equations(y)]
    xrhs = [xi.rhs for xi in equations(x)]
    @inbounds for (i, xi) in enumerate(xrhs)
        n[i] = any(isequal.([xi], yrhs))
        !n[i] && break
    end
    return all(n)
end

free_parameters(b::AbstractBasis; operations = [+]) = count_operation([xi.rhs for xi in b.eqs], operations) + length(b.eqs)
