import Base.==

mutable struct Basis{B, V, P, T} <: AbstractBasis
    """The equations of the basis"""
    basis::B
    """Dependent (state) variables"""
    variables::V
    """Parameter variables"""
    parameter::P
    """Independent variable"""
    iv::T
    """Internal function representation of the basis field"""
    f_::Function
end

Base.show(io::IO, x::Basis) = print(io, "$(length(x.basis)) dimensional basis in ", "$(String.([v.op.name for v in x.variables]))")

@inline function Base.print(io::IO, x::Basis)
    show(io, x)
    println(io)
    if length(x.variables) == length(x.basis)
        for (i, bi) in enumerate(x.basis)
            println(io,"d$(x.variables[i]) = $bi")
        end
    else
        for (i, bi) in enumerate(x.basis)
            println(io,"f_$i = $bi")
        end
    end
end

is_independent(o::Operation) = isempty(o.args)



"""
    Basis(f, u; p, iv, linear_independent = false, simplify_eqs = true, eval_expression = false)

A basis over the variables `u` with parameters `p` and independent variable `iv`.
`f` can either be a Julia function which is able to use ModelingToolkit variables or
a vector of `Operation`.
It can be called with the typical DiffEq signature, meaning out of place with `f(u,p,t)`
or in place with `f(du, u, p, t)`.
If `linear_independent` is set to `true`, a linear independent basis is created from all atom function in `f`.
If `simplify_eqs` is set to `true`, `simplify` is called on `f`.

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
function Basis(basis::AbstractArray{Operation}, variables::AbstractArray{Operation};
               parameters::AbstractArray =  Operation[], iv = nothing, linear_independent::Bool = false, simplify_eqs = true, eval_expression = false)
    @assert all(is_independent.(variables)) "Please provide independent states."

    bs = deepcopy(basis)
    simplify_eqs && (bs = simplify.(bs))
    linear_independent && (bs = create_linear_independent_eqs(bs))
    unique!(bs)

    if isnothing(iv)
        @parameters t
        iv = t
    end

    vs = [ModelingToolkit.Variable(Symbol(i)) for i in variables]
    ps = [ModelingToolkit.Variable(Symbol(i)) for i in parameters]

    if eval_expression
        f_oop, f_iip = eval.(ModelingToolkit.build_function(bs, vs, ps, [iv], expression = Val{true}))
    else
        f_oop, f_iip = ModelingToolkit.build_function(bs, vs, ps, [iv], expression = Val{false})
    end

    f_(u, p, t) = f_oop(u, p, t)
    f_(du, u, p, t) = f_iip(du, u, p, t)

    return Basis(bs, variables, parameters, iv, f_)
end


function Basis(basis::Function, variables::AbstractArray{Operation};  parameters::AbstractArray =  Operation[], iv = nothing, kwargs...)
    @assert all(is_independent.(variables)) "Please provide independent variables for basis."

    if isnothing(iv)
        @parameters t
        iv = t
    end

    try
        eqs = basis(variables, parameters, iv)
        return Basis(eqs, variables, parameters = parameters, iv = iv, kwargs...)
    catch e
        rethrow(e)
    end
end


function update!(basis::Basis,eval_expression = false)

    vs = [ModelingToolkit.Variable(Symbol(i))(basis.iv) for i in variables(basis)]
    ps = [ModelingToolkit.Variable(Symbol(i)) for i in parameters(basis)]

    if eval_expression
        f_oop, f_iip = eval.(ModelingToolkit.build_function(basis.basis, vs, ps, [basis.iv], expression = Val{false}))
    else
        f_oop, f_iip = ModelingToolkit.build_function(basis.basis, vs, ps, [basis.iv], expression = Val{false})
    end

    f_(u, p, t) = f_oop(u, p, t)
    f_(du, u, p, t) = f_iip(du, u, p, t)

    basis.f_ = f_
    return
end


"""
    push!(basis, op)

    Push the operation(s) in `op` into the basis and update all internal fields accordingly.
    `op` can either be a single `Operation` or an Array of `Operation`s.
"""
function Base.push!(b::Basis, ops::AbstractArray{Operation}, unique::Bool = true)
    @inbounds for o in ops
        push!(b.basis, o, false)
    end
    unique ? unique!(b.basis) : nothing
    update!(b)
    return
end

function Base.push!(b::Basis, op₀::Operation, unique::Bool = true)
    op = simplify(op₀)
    push!(b.basis, op)
    # Check for uniqueness
    unique ? unique!(b) : nothing
    update!(b)
    return
end

"""
    deleteat!(basis, inds)

    Delete the entries specified by `inds` and update the `Basis` accordingly.
"""
function Base.deleteat!(b::Basis, inds)
    deleteat!(b.basis, inds)
    update!(b)
    return
end

"""
    merge(basis_a, basis_b)

    Return a new `Basis`, which is defined via the union of both bases.
"""
function Base.merge(basis_a::Basis, basis_b::Basis)
    b =  unique(vcat(basis_a.basis, basis_b.basis))
    vs = unique(vcat(basis_a.variables, basis_b.variables))
    ps = unique(vcat(basis_a.parameter, basis_b.parameter))
    return Basis(b, vs, parameters = ps)
end

"""
    merge!(basis_a, basis_b)

    Updates the `Basis` to include the union of both bases.
"""
function Base.merge!(basis_a::Basis, basis_b::Basis)
    push!(basis_a, basis_b.basis)
    basis_a.variables = unique(vcat(basis_a.variables, basis_b.variables))
    basis_a.parameter = unique(vcat(basis_a.parameter, basis_b.parameter))
    update!(basis_a)
    return
end

Base.getindex(b::Basis, idx) = b.basis[idx]
Base.firstindex(b::Basis) = firstindex(b.basis)
Base.lastindex(b::Basis) = lastindex(b.basis)
Base.iterate(b::Basis) = iterate(b.basis)
Base.iterate(b::Basis, id) = iterate(b.basis, id)

function (==)(x::Basis, y::Basis)
    n = zeros(Bool, length(x.basis))
    @inbounds for (i, xi) in enumerate(x)
        n[i] = any(isequal.(xi, y.basis))
    end
    return all(n)
end

function is_unary(f::Function)
    for m in methods(f)
        m.nargs - 1 > 1 && return false
    end
    return true
end

function types_in_f_tuple(sig)
    sig.parameters[2:end]
end

function types_in_f_tuple(sig::UnionAll)
    Base.unwrap_unionall(sig).parameters[2:end]    
end

function is_unary(f::Function, dtype::DataType)
    for m in methods(f)
        if dtype ∈ types_in_f_tuple(m.sig)
            m.nargs - 1 > 1 && return false
        end

    end
    return true
end

function count_operation(x::T, op::Function, nested::Bool = true) where T <: Expression
    isa(x, ModelingToolkit.Constant) && return 0
    isa(x.op, Expression) && return 0
    if x.op == op
        if is_unary(op)
            # Handles sin, cos and stuff
            nested && return 1 + count_operation(x.args, op)
            return 1
        else
            # Handles +, *
            nested && length(x.args)-1 + count_operation(x.args, op) 
            return length(x.args)-1 
        end
    elseif nested
        return count_operation(x.args, op, nested)
    end
    return 0
end

function count_operation(x::T, ops::AbstractArray,nested::Bool = true) where T<:Expression
    c_ops = 0
    for oi in ops
        c_ops += count_operation(x, oi, nested)
    end
    return c_ops
end

function count_operation(x::AbstractVector{T}, op, nested::Bool = true) where T <: Expression
    c_ops = 0
    for xi in x
        c_ops += count_operation(xi, op, nested)
    end
    return c_ops
end

function remove_constant_factor(o::T) where T <: Expression
    isa(o, ModelingToolkit.Constant) && return ModelingToolkit.Constant(1)
    n_ops = count_operation(o, *, false) +1
    ops = Array{Expression}(undef, n_ops)
    @views split_operation!(ops, o, [*])
    filter!(x->!isa(x, ModelingToolkit.Constant), ops)
    return prod(ops)
end

function remove_constant_factor!(o::AbstractArray{T}) where T <: Expression
    for i in eachindex(o)
        o[i] = remove_constant_factor(o[i])
    end
end

function split_operation!(k::AbstractVector{T}, o::Expression, ops::AbstractArray = [+]) where T <: Expression
    n_ops = count_operation(o, ops, false) 
    c_ops = 0
    @views begin
        if n_ops == 0
            k .= o
        else
            counter_ = 1
            for oi in o.args
                c_ops = count_operation(oi, ops, false)
                split_operation!(k[counter_:counter_+c_ops], oi, ops)
                counter_ += c_ops + 1
            end
        end
    end
end

function create_linear_independent_eqs(o::AbstractVector{T}) where T <: Expression
    unique!(o)
    n_ops = [count_operation(bi, +, false) for bi in o]
    n_x = sum(n_ops) + length(o)
    u_o = Array{T}(undef, n_x)
    ind_lo, ind_up = 0, 0
    for i in eachindex(o)
        ind_lo = i > 1 ? sum(n_ops[1:i-1]) + i : 1
        ind_up = sum(n_ops[1:i]) + i
        @views split_operation!(u_o[ind_lo:ind_up], o[i], [+])
    end
    remove_constant_factor!(u_o)
    unique!(u_o)
    return u_o
end

free_parameters(b::Basis; operations = [+]) = count_operation(b.basis, operations) + length(b.basis)

(b::Basis)(u, p::DiffEqBase.NullParameters, t) = b(u, [], t)
(b::Basis)(du, u, p::DiffEqBase.NullParameters, t) = b(du, u, [], t)
(b::Basis)(u::AbstractVector,  p::AbstractArray = [], t = nothing) = b.f_(u, isempty(p) ? parameters(b) : p, isnothing(t) ? zero(eltype(u)) : t)
(b::Basis)(du::AbstractVector, u::AbstractVector, p::AbstractArray = [], t = nothing) = b.f_(du, u, isempty(p) ? parameters(b) : p, isnothing(t) ? zero(eltype(u)) : t)

function (b::Basis)(x::AbstractMatrix, p::AbstractArray = [], t::AbstractArray = [])
    isempty(t) ? nothing : @assert size(x, 2) == length(t)

    if (isempty(p) || eltype(p) <: Expression) && !isempty(parameters(b))
        pi = isempty(p) ? parameters(b) : p
        res = zeros(eltype(pi), length(b), size(x)[2])
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

Base.size(b::Basis) = size(b.basis)
Base.length(b::Basis) = length(b.basis)

"""
    parameters(basis)

    Returns the parameters of the basis.
"""
ModelingToolkit.parameters(b::Basis) = b.parameter

"""
    variables(basis)

    Returns the dependent variables of the basis.
"""
variables(b::Basis) = b.variables

"""
    independent_variable(basis)

    Returns the independent_variable of the basis.
"""
ModelingToolkit.independent_variable(b::Basis) = b.iv


"""
    jacobian(basis)

    Returns a function representing the jacobian matrix / gradient of the `Basis` with respect to the
    dependent variables as a function with the common signature `f(u,p,t)` for out of place and `f(du, u, p, t)` for in place computation.
"""
function jacobian(basis::Basis, eval_expression = false)

    vs = [ModelingToolkit.Variable(Symbol(i))(independent_variable(basis)) for i in variables(basis)]
    ps = [ModelingToolkit.Variable(Symbol(i)) for i in parameters(basis)]

    j = ModelingToolkit.jacobian(basis.basis, variables(basis))

    if eval_expression
        f_oop, f_iip = eval.(ModelingToolkit.build_function(expand_derivatives.(j), vs, ps, [basis.iv], expression = Val{true}))
    else
        f_oop, f_iip = ModelingToolkit.build_function(expand_derivatives.(j), vs, ps, [basis.iv], expression = Val{false})
    end

    f_(u, p, t) = f_oop(u, p, t)
    f_(du, u, p, t) = f_iip(du, u, p, t)

    return f_
end


function Base.unique!(b::Basis)
    N = length(b.basis)
    removes = Vector{Bool}()
    for i ∈ 1:N
        push!(removes, any([isequal(b.basis[i], b.basis[j]) for j in i+1:N]))
    end
    deleteat!(b, removes)
    update!(b)
end

function Base.unique(b::Basis)
    N = length(b.basis)
    returns = Vector{Bool}()
    for i ∈ 1:N
        push!(returns, any([isequal(b.basis[i], b.basis[j]) for j in i+1:N]))
    end
    returns = [!r for r in returns]
    return Basis(b.basis[returns], variables(b), parameters = parameters(b))
end

function Base.unique(b₀::AbstractVector{Operation})
    b = simplify.(b₀)
    N = length(b)
    returns = Vector{Bool}()
    for i ∈ 1:N
        push!(returns, any([isequal(b[i], b[j]) for j in i+1:N]))
    end
    returns = [!r for r in returns]
    return b[returns]
end

function Base.unique!(b::AbstractArray{Operation})
    N = length(b)
    removes = Vector{Bool}()
    for i ∈ 1:N
        push!(removes, any([isequal(b[i], b[j]) for j in i+1:N]))
    end
    deleteat!(b, removes)
end

"""
    dynamics(basis)

    Returns the internal function representing the dynamics of the `Basis`.
"""
function dynamics(b::Basis)
    return b.f_
end

"""
    ODESystem(basis)

    Converts the `Basis` into an `ODESystem` defined via `ModelingToolkit.jl`.
"""
function ModelingToolkit.ODESystem(b::Basis)
    @assert length(b) == length(variables(b))
    # Define the time
    @derivatives D'~independent_variable(b)

    vs = similar(b.variables)
    dvs = similar(b.variables)

    for (i, vi) in enumerate(b.variables)
        vs[i] = ModelingToolkit.Operation(vi.op, [independent_variable(b)])
        dvs[i] = D(vs[i])
    end
    eqs = dvs .~ b(vs, parameters(b), independent_variable(b))
    return ODESystem(eqs, independent_variable(b), variables(b), parameters(b))
end
