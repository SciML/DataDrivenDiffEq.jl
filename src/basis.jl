using ModelingToolkit
using LinearAlgebra
using DiffEqBase
import Base.==
import Base.unique, Base.unique!
using ModelingToolkit: <ₑ, value, isparameter

"""
$(TYPEDEF)

A basis over the variables `u` with parameters `p` and independent variable `iv`.
It extends an `AbstractSystem` as defined in `ModelingToolkit.jl`.
`f` can either be a Julia function which is able to use ModelingToolkit variables or
a vector of `eqs`.
It can be called with the typical DiffEq signature, meaning out of place with `f(u,p,t)`
or in place with `f(du, u, p, t)`.
If `linear_independent` is set to `true`, a linear independent basis is created from all atom function in `f`.
If `simplify_eqs` is set to `true`, `simplify` is called on `f`.
Additional keyworded arguments include `name`, which can be used to name the basis, `pins` used for connections and
`observed` for defining observeables. 

# Fields
$(FIELDS)

# Example

```julia
using ModelingToolkit
using DataDrivenDiffEq

@parameters w[1:2] t
@variables u[1:2]

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
mutable struct Basis <: ModelingToolkit.AbstractSystem
    """The equations of the basis"""
    eqs::Vector{Equation}
    """Dependent (state) variables"""
    states::Vector
    """Parameters"""
    ps::Vector
    pins::Vector
    observed::Vector
    """Independent variable"""
    iv::Num
    """Internal function representation of the basis"""
    f_::Function
    """Name of the basis"""
    name::Symbol
    """Internal systems"""
    systems::Vector{Basis}
end

is_independent(t::Term) = isempty(t.args)
is_independent(s::Sym) = true
is_independent(x::Num) = is_independent(ModelingToolkit.value(x))

function Basis(eqs::AbstractVector, states::AbstractVector; parameters::AbstractArray = [], iv = nothing,
    simplify = false, linear_independent = false, name = gensym(:Basis), eval_expression = false,
    pins = [], observed = [],
    kwargs...)
    @assert all(is_independent.(states)) "Please provide independent states."

    eqs = simplify ? ModelingToolkit.simplify.(eqs) : eqs
    eqs = linear_independent ? create_linear_independent_eqs(eqs) : eqs
    isnothing(iv) && (iv = Num(Variable(:t)))
    unique!(eqs, !simplify)
    
    if eval_expression
        f_oop, f_iip = eval.(build_function(eqs, value.(states), value.(parameters), [value(iv)], expression = Val{true}))
    else 
        f_oop, f_iip = build_function(eqs, value.(states), value.(parameters), [value(iv)], expression = Val{false})
    end
    eqs = [Variable(:φ,i) ~ eq for (i,eq) ∈ enumerate(eqs)]
    f_(u,p,t) = f_oop(u,p,t)
    f_(du, u, p, t) = f_iip(du, u, p, t)

    return Basis(eqs, value.(states), value.(parameters), pins, observed, value(iv), f_, name, Basis[])
end

function Basis(f::Function, states::AbstractVector; parameters::AbstractArray = [], iv = nothing, kwargs...)
    @assert all(is_independent.(states)) "Please provide independent states."
    
    isnothing(iv) && (iv = Num(Variable(:t)))
    try
        eqs = f(states, parameters, iv)
        return Basis(eqs, states, parameters = parameters, iv = iv; kwargs...)
    catch e
        rethrow(e)
    end
end

Base.show(io::IO, x::Basis) = print(io, "$(String.(x.name)) : $(length(x.eqs)) dimensional basis in ", "$(String.([value(v).name for v in x.states]))")

@inline function Base.print(io::IO, x::Basis)
    show(io, x)
    !isempty(x.ps) && println(io, "\nParameters : $(x.ps)")
    println(io, "\nIndependent variable: $(x.iv)")
    println(io, "Equations")
    for (i,eq) ∈ enumerate(x.eqs)
        if i < 5 || i == length(x.eqs)
        println(io, "$(eq.lhs) = $(eq.rhs)")
        elseif i == 5
            println(io, "...")
        else 
            continue
        end
    end
end

@inline function Base.println(io::IO, x::Basis, fullview::DataType = Val{false})
    fullview == Val{false} && return print(io, x) 
    show(io, x)
    !isempty(x.ps) && println(io, "\nParameters : $(x.ps)")
    println(io, "\nIndependent variable: $(x.iv)")
    println(io, "Equations")
    for (i,eq) ∈ enumerate(x.eqs)
        println(io, "$(eq.lhs) = $(eq.rhs)")
    end
end

"""

"""
variables(x::Basis) = x.states

function update!(b::Basis, eval_expression = false)

    if eval_expression
        f_oop, f_iip = eval.(ModelingToolkit.build_function([bi.rhs for bi in b.eqs], b.states, b.ps, [b.iv], expression = Val{true}))
    else
        f_oop, f_iip = ModelingToolkit.build_function([bi.rhs for bi in b.eqs], b.states, b.ps, [b.iv], expression = Val{false})
    end

    f_(u, p, t) = f_oop(u, p, t)
    f_(du, u, p, t) = f_iip(du, u, p, t)

    b.f_ = f_
    return
end

function unique(b::AbstractArray{Num}, simplify_eqs = false)
    b = simplify_eqs ? simplify.(b) : b
    returns = ones(Bool, size(b)...)
    N = maximum(eachindex(b))
    for i ∈ eachindex(b)
        returns[i] = !any([isequal(b[i], b[j]) for j in i+1:N])
    end
    return b[returns]
end

function Base.unique!(b::AbstractArray, simplify_eqs = false)
    bs = simplify_eqs ? simplify.(b) : b
    removes = zeros(Bool, size(bs)...)
    N = maximum(eachindex(bs))
    for i ∈ eachindex(bs)
        removes[i] = any([isequal(bs[i], bs[j]) for j in i+1:N])
    end
    deleteat!(b, removes)
end

function unique(b::AbstractArray{Equation}, simplify_eqs = false)
    b = simplify_eqs ? simplify.(b) : b
    returns = ones(Bool, size(b)...)
    N = maximum(eachindex(b))
    for i ∈ eachindex(b)
        returns[i] = !any([isequal(b[i].rhs, b[j].rhs) for j in i+1:N])
    end
    return b[returns]
end

function Base.unique!(b::AbstractArray{Equation}, simplify_eqs = false)
    bs = [bi.rhs for bi in b]
    bs = simplify_eqs ? simplify.(bs) : bs
    removes = zeros(Bool, size(bs)...)
    N = maximum(eachindex(bs))
    for i ∈ eachindex(bs)
        removes[i] = any([isequal(bs[i], bs[j]) for j in i+1:N])
    end
    deleteat!(b, removes)
end


function Base.unique!(b::Basis, simplify_eqs = false; eval_expression = false)
    unique!(b.eqs, simplify_eqs)
    update!(b, eval_expression)
end

function Base.unique(b::Basis; kwargs...)
    eqs = unique(map(x->x.rhs, equations(b)))
    return Basis(eqs, variables(b), parameters = parameters(b), iv = independent_variable(b), kwargs...)
end

"""
    deleteat!(basis, inds, eval_expression = false)

    Delete the entries specified by `inds` and update the `Basis` accordingly.
"""
function Base.deleteat!(b::Basis, inds; eval_expression = false)
    deleteat!(b.eqs, inds)
    update!(b, eval_expression)
    return
end

"""
    push!(basis, eq, simplify_eqs = true; eval_expression = false)

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
    push!(b.eqs, Variable(:φ, length(b.eqs)+1)~eq)
    unique!(b, simplify_eqs, eval_expression = eval_expression)
    return
end

"""
    merge(x::Basis, y::Basis; eval_expression = false)

    Return a new `Basis`, which is defined via the union of `x` and `y` .
"""
function Base.merge(x::Basis, y::Basis; eval_expression = false)
    b =  unique(vcat([xi.rhs  for xi ∈ equations(x)], [xi.rhs  for xi ∈ equations(y)]))
    vs = unique(vcat(x.states, y.states))
    ps = unique(vcat(x.ps, y.ps))
    pins = unique(vcat(x.pins, y.pins))
    observed = unique(vcat(x.observed, y.observed))
    return Basis(Num.(b), vs, parameters = ps, pins = pins, observed = observed, eval_expression = eval_expression)
end

"""
    merge!(x::Basis, y::Basis; eval_expression = false)

    Updates `x` to include the union of both `x` and `y`.
"""
function Base.merge!(x::Basis, y::Basis; eval_expression = false)
    push!(x, map(x->x.rhs, equations(y)))
    x.states = unique(vcat(x.states, y.states))
    x.ps = unique(vcat(x.ps, y.ps))
    update!(x, eval_expression)
    return
end

Base.length(x::Basis) = length(x.eqs)
Base.size(x::Basis) = size(x.eqs)

Base.getindex(x::Basis, idx) = getindex(equations(x), idx)
Base.firstindex(x::Basis) = firstindex(equations(x))
Base.lastindex(x::Basis) = lastindex(equations(x))
Base.iterate(x::Basis) = iterate(equations(x))
Base.iterate(x::Basis, id) = iterate(equations(x), id)

function (==)(x::Basis, y::Basis)
    length(x) == length(y) || return false
    n = zeros(Bool, length(x))
    yrhs = [yi.rhs for yi in equations(y)]
    xrhs = [xi.rhs for xi in equations(x)]
    @inbounds for (i, xi) in enumerate(xrhs)
        n[i] = any(isequal.([xi], yrhs))
    end
    return all(n)
end

free_parameters(b::Basis; operations = [+]) = count_operation([xi.rhs for xi in b.eqs], operations) + length(b.eqs)

(b::Basis)(u, p::DiffEqBase.NullParameters, t) = b(u, [], t)
(b::Basis)(du, u, p::DiffEqBase.NullParameters, t) = b(du, u, [], t)
(b::Basis)(u::AbstractVector,  p::AbstractArray = [], t = nothing) = b.f_(u, isempty(p) ? parameters(b) : p, isnothing(t) ? zero(eltype(u)) : t)
(b::Basis)(du::AbstractVector, u::AbstractVector, p::AbstractArray = [], t = nothing) = b.f_(du, u, isempty(p) ? parameters(b) : p, isnothing(t) ? zero(eltype(u)) : t)

function (b::Basis)(x::AbstractMatrix, p::AbstractArray = [], t::AbstractArray = [])
    isempty(t) ? nothing : @assert size(x, 2) == length(t)

    if (isempty(p) || eltype(p) <: Num) && !isempty(parameters(b))
        pi = isempty(p) ? parameters(b) : p
        res = Array{Any}(undef,length(b), size(x)[2])
    else
        pi = p
        res = zeros(eltype(x), length(b), size(x)[2])
    end

    @inbounds for i in 1:size(x)[2]
        res[:, i] .= b.f_(x[:, i], isempty(p) ? parameters(b) : p, isempty(t) ? zero(eltype(x)) : t[i])
    end
    return res
end

function (b::Basis)(y::AbstractMatrix, x::AbstractMatrix, p::AbstractArray = [], t::AbstractArray = [])
    @assert size(x, 2) == size(y, 2) "Provide consistent arrays."
    @assert size(y, 1) == length(b) "Provide consistent arrays."
    isempty(t) ? nothing : @assert size(x, 2) == length(t)

    @inbounds for i in 1:size(x, 2)
        b.f_(view(y, :, i), view(x, :, i), isempty(p) ? parameters(b) : p, isempty(t) ? zero(eltype(x)) : t[i])
    end
end


"""
    jacobian(basis)

    Returns a function representing the jacobian matrix / gradient of the `Basis` with respect to the
    dependent variables as a function with the common signature `f(u,p,t)` for out of place and `f(du, u, p, t)` for in place computation.
"""
function jacobian(x::Basis, eval_expression = false)

    j = ModelingToolkit.jacobian([xi.rhs for xi in x.eqs], x.states)

    if eval_expression
        f_oop, f_iip = eval.(ModelingToolkit.build_function(expand_derivatives.(j), x.states, x.ps, [x.iv], expression = Val{true}))
    else
        f_oop, f_iip = ModelingToolkit.build_function(expand_derivatives.(j), x.states, x.ps, [x.iv], expression = Val{false})
    end

    f_(u, p, t) = f_oop(u, p, t)
    f_(du, u, p, t) = f_iip(du, u, p, t)

    return f_
end


"""
    dynamics(basis)

    Returns the internal function representing the dynamics of the `Basis`.
"""
function dynamics(b::Basis)
    return b.f_
end

## Term Manipulation

function is_unary(f::Function)
    for m in methods(f)
        m.nargs - 1 > 1 && return false
    end
    return true
end

count_operation(x::Number, op::Function, nested::Bool = true) = 0
count_operation(x::Sym, op::Function, nested::Bool = true) = 0
count_operation(x::Num, op::Function, nested::Bool = true) = count_operation(value(x), op, nested)

function count_operation(x::Term, op::Function, nested::Bool = true)
    if x.f == op
        if is_unary(op)
            # Handles sin, cos and stuff
            nested && return 1 + count_operation(x.arguments, op)
            return 1
        else
            # Handles +, *
            nested && length(x.arguments)-1 + count_operation(x.arguments, op) 
            return length(x.arguments)-1 
        end
    elseif nested
        return count_operation(x.arguments, op, nested)
    end
    return 0
end

function count_operation(x, ops::AbstractArray, nested::Bool = true)
    return sum([count_operation(x, op, nested) for op in ops])
end

function count_operation(x::AbstractArray, op::Function, nested::Bool = true)
    sum([count_operation(xi, op, nested) for xi in x])
end

function count_operation(x::AbstractArray, ops::AbstractArray, nested::Bool = true)
    counter = 0
    @inbounds for xi in x, op in ops
        counter += count_operation(xi, op, nested)
    end
    counter
end

function split_term!(x::AbstractArray, o::Term, ops::AbstractArray = [+])
    n_ops = count_operation(o, ops, false) 
    c_ops = 0
    @views begin
        if n_ops == 0
            x[begin]= o
        else
            counter_ = 1
            for oi in o.arguments
                c_ops = count_operation(oi, ops, false)
                split_term!(x[counter_:counter_+c_ops], oi, ops)
                counter_ += c_ops + 1
            end
        end
    end
end

split_term!(x::AbstractArray,o::Num, ops::AbstractArray = [+]) = split_term!(x, value(o), ops)

function split_term!(x::AbstractArray, o::Sym, ops::AbstractArray = [+]) 
    x[begin] = o
    return
end

function split_term!(x::AbstractArray, o::Number, ops::AbstractArray = [+]) 
    x[begin] = o
    return
end

remove_constant_factor(x::Num) = remove_constant_factor(value(x))
remove_constant_factor(x::Sym) = x
remove_constant_factor(x::Number) = one(x)

function remove_constant_factor(x::Term)
    n_ops = count_operation(x, *, false)+1
    ops = Array{Any}(undef, n_ops)
    @views split_term!(ops, x, [*])
    filter!(x->!isa(x, Number), ops)
    return Num(prod(ops))
end

function remove_constant_factor!(o::AbstractArray)
    for i in eachindex(o)
        o[i] = remove_constant_factor(o[i])
    end
end

function create_linear_independent_eqs(o::AbstractVector)
    o .= simplify.(o)
    remove_constant_factor!(o)
    n_ops = [count_operation(bi, +, false) for bi in o]
    n_x = sum(n_ops) + length(o)
    u_o = Array{Any}(undef, n_x)
    ind_lo, ind_up = 0, 0
    for i in eachindex(o)
        ind_lo = i > 1 ? sum(n_ops[1:i-1]) + i : 1
        ind_up = sum(n_ops[1:i]) + i
        @views split_term!(u_o[ind_lo:ind_up], o[i], [+])
    end
    remove_constant_factor!(u_o)
    unique!(u_o)
    return u_o
end
