
## Utilities

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
    push!(equations(b), Variable(:φ, length(b.eqs)+1)~eq)
    unique!(b, simplify_eqs, eval_expression = eval_expression)
    return
end

function Base.push!(b::Basis, eq::Equation, simplify_eqs = true; eval_expression = false)
    push!(equations(b), eq)
    unique!(b, simplify_eqs, eval_expression = eval_expression)
    return
end


"""
    merge(x::Basis, y::Basis; eval_expression = false)

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
    merge!(x::Basis, y::Basis; eval_expression = false)

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

##

## Get unary and binary functions

function is_unary(f::Function, t::Type = Number)
    f ∈ [+, -, *, /, ^] && return false
    for m in methods(f, (t, ))
        m.nargs - 1 > 1 && return false
    end
    return true
end

function is_binary(f::Function, t::Type = Number)
    f ∈ [+, -, *, /, ^] && return true
    !is_unary(f, t)
end

function ariety(f::Function, t::Type = Number)
    is_unary(f, t) && return 1
    is_binary(f, t) && return 2
    return 0
end

function sort_ops(f::Vector{Function})
    U = Function[]
    B = Function[]
    for fi in f
        is_unary(fi) ? push!(U, fi) : push!(B, fi)
    end
    return U, B
end

## Create linear independent basis

count_operation(x::Number, op::Function, nested::Bool = true) = 0
count_operation(x::Sym, op::Function, nested::Bool = true) = 0
count_operation(x::Num, op::Function, nested::Bool = true) = count_operation(value(x), op, nested)

function count_operation(x, op::Function, nested::Bool = true)
    if operation(x)== op
        if is_unary(op)
            # Handles sin, cos and stuff
            nested && return 1 + count_operation(arguments(x), op)
            return 1
        else
            # Handles +, *
            nested && length(arguments(x))-1 + count_operation(arguments(x), op)
            return length(arguments(x))-1
        end
    elseif nested
        return count_operation(arguments(x), op, nested)
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

function split_term!(x::AbstractArray, o, ops::AbstractArray = [+])
    if istree(o)
        n_ops = count_operation(o, ops, false)
        c_ops = 0
        @views begin
            if n_ops == 0
                x[begin]= o
            else
                counter_ = 1
                for oi in arguments(o)
                    c_ops = count_operation(oi, ops, false)
                    split_term!(x[counter_:counter_+c_ops], oi, ops)
                    counter_ += c_ops + 1
                end
            end
        end
    else
        x[begin] = o
    end
    return
end

split_term!(x::AbstractArray,o::Num, ops::AbstractArray = [+]) = split_term!(x, value(o), ops)

remove_constant_factor(x::Num) = remove_constant_factor(value(x))
remove_constant_factor(x::Number) = one(x)

function remove_constant_factor(x)
    # Return, if the function is nested
    istree(x) || return x
    # Count the number of operations
    n_ops = count_operation(x, [*], false)+1
    # Create a new array
    ops = Array{Any}(undef, n_ops)
    @views split_term!(ops, x, [*])
    filter!(x->!isa(x, Number), ops)
    return Num(prod(ops))
end

function remove_constant_factor(o::AbstractArray)
    oi = Array{Any}(undef, size(o))
    for i in eachindex(o)
        oi[i] = remove_constant_factor(o[i])
    end
    return Num.(oi)
end

function create_linear_independent_eqs(ops::AbstractVector, simplify_eqs::Bool = false)
    o = simplify.(ops)
    o = remove_constant_factor(o)
    n_ops = [count_operation(oi, +, false) for oi in o]
    n_x = sum(n_ops) + length(o)
    u_o = Array{Any}(undef, n_x)
    ind_lo, ind_up = 0, 0
    for i in eachindex(o)
        ind_lo = i > 1 ? sum(n_ops[1:i-1]) + i : 1
        ind_up = sum(n_ops[1:i]) + i

        @views split_term!(u_o[ind_lo:ind_up], o[i], [+])
    end
    u_o = remove_constant_factor(u_o)
    unique!(u_o)
    return simplify_eqs ? simplify.(Num.(u_o)) : Num.(u_o)
end
