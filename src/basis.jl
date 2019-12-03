mutable struct Basis{O, V, P} <: abstractBasis
    basis::O
    variables::V
    parameter::P
    f_
end

is_independent(o::Operation) = isempty(o.args)

@inline function Base.print(io::IO, x::Basis)
    show(io, x)
    println()
    for (i, bi) in enumerate(x.basis)
        println("d$(x.variables[i]) = $bi")
    end
end

Base.show(io::IO, x::Basis) = print(io, "$(length(x.basis)) dimensional basis in ", "$(String.([v.op.name for v in x.variables]))")

function Basis(basis::AbstractVector{Operation}, variables::AbstractVector{Operation};  parameters =  [])
    @assert all(is_independent.(variables)) "Please provide independent variables for base."

    bs = unique(basis)

    vs = sort!([b for b in [ModelingToolkit.vars(bs)...] if !b.known], by = x -> x.name)
    ps = sort!([b for b in [ModelingToolkit.vars(bs)...] if b.known], by = x -> x.name )

    f_ = ModelingToolkit.build_function(bs, vs, ps, (), simplified_expr, Val{false})[1]
    return Basis(bs, variables, parameters, f_)
end

function update!(b::Basis)
    vs = sort!([bi for bi in [ModelingToolkit.vars(b.basis)...] if !bi.known], by = x->x.name)
    ps = sort!([bi for bi in [ModelingToolkit.vars(b.basis)...] if bi.known], by = x->x.name)

    b.f_ = ModelingToolkit.build_function(b.basis, vs, ps, (), simplified_expr, Val{false})[1]
    return
end

function Base.push!(b::Basis, op::Operation)
    push!(b.basis, op)
    # Check for uniqueness
    unique!(b)
    update!(b)
    return
end

function Base.deleteat!(b::Basis, inds)
    deleteat!(b.basis, inds)
    update!(b)
    return
end

(b::Basis)(x::AbstractArray; p::AbstractArray = []) = b.f_(x, p)

Base.size(b::Basis) = size(b.basis)
ModelingToolkit.parameters(b::Basis) = b.parameter
variables(b::Basis) = b.variables

function jacobian(b::Basis)
    vs = sort!([bi for bi in [ModelingToolkit.vars(b.basis)...] if !bi.known], by = x-> x.name)
    ps = sort!([bi for bi in [ModelingToolkit.vars(b.basis)...] if bi.known], by = x-> x.name)
    j = calculate_jacobian(b.basis, variables(b))
    return ModelingToolkit.build_function(expand_derivatives.(j), vs, ps, (), simplified_expr, Val{false})[1]
end

function Base.unique!(b::Basis)
    N = length(b.basis)
    removes = Vector{Bool}()
    for i ∈ 1:N
        push!(removes, any([isequal(b.basis[i], b.basis[j]) for j in i+1:N]))
    end
    deleteat!(b, removes)
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

function dynamics(b::Basis)
    return b.f_
end

function ModelingToolkit.ODESystem(b::Basis)
    # Define the time
    @parameters t
    @derivatives D'~t

    vs = similar(b.variables)
    dvs = similar(b.variables)
    for (i, vi) in enumerate(b.variables)
        vs[i] = ModelingToolkit.Operation(vi.op, [t])
        dvs[i] = D(vs[i])
    end
    eqs = dvs .~ b(vs, p = b.parameter)
    return ODESystem(eqs)
end

function count_operations(o::Expression, ops::AbstractArray)
    isa(o, ModelingToolkit.Constant) || return 0
    k = o.op ∈ ops ? 1 : 0
    if !isempty(o.args)
        k += sum([count_operations(ai, ops) for ai in o.args])
    end
    return k
end

free_parameters(b::Basis; operations = [+]) = sum([count_operations(bi, operations) for bi in b.basis]) + length(b.basis)
