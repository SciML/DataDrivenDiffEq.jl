mutable struct Basis{O, V, P} <: abstractBasis
    basis::O
    variables::V
    parameter::P
    f_
end

is_independent(o::Operation) = isempty(o.args)

Base.show(io::IO, x::Basis) = print(io, "$(length(x.basis)) dimensional basis in ", "$(String.([v.op.name for v in x.variables]))")
@inline function Base.print(io::IO, x::Basis)
    show(io, x)
    println()
    for (i, bi) in enumerate(x.basis)
        println("d$(x.variables[i]) = $bi")
    end
end

function Basis(basis::AbstractVector{Operation}, variables::AbstractVector{Operation};  parameters =  [])
    @assert all(is_independent.(variables)) "Please provide independent variables for base."

    bs = unique(basis)
    fix_single_vars_in_basis!(bs, variables)

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

function Base.push!(b::Basis, ops::AbstractArray{Operation})
    @inbounds for o in ops
        push!(b.basis, o)
    end
    unique!(b.basis)
    update!(b)
    return
end

function Base.push!(b::Basis, op₀::Operation)
    op = simplify_constants(op₀)
    fix_single_vars_in_basis!(op, b.variables)
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

function Base.merge(basis_a::Basis, basis_b::Basis)
    b =  unique(vcat(basis_a.basis, basis_b.basis))
    vs = unique(vcat(basis_a.variables, basis_b.variables))
    ps = unique(vcat(basis_a.parameter, basis_b.parameter))
    return Basis(b, vs, parameters = ps)
end

function Base.merge!(basis_a::Basis, basis_b::Basis)
    push!(basis_a, basis_b.basis)
    basis_a.variables = unique(vcat(basis_a.variables, basis_b.variables))
    basis_b.variables = unique(vcat(basis_a.parameter, basis_b.parameter))
    return
end

function count_operation(o::Expression, ops::AbstractArray)
    if isa(o, ModelingToolkit.Constant)
        return 0
    end
    k = o.op ∈ ops ? 1 : 0
    if !isempty(o.args)
        k += sum([count_operation(ai, ops) for ai in o.args])
    end
    return k
end

free_parameters(b::Basis; operations = [+]) = sum([count_operation(bi, operations) for bi in b.basis]) + length(b.basis)

(b::Basis)(x::AbstractArray{T, 1}; p::AbstractArray = []) where T <: Number = b.f_(x, p)

function (b::Basis)(x::AbstractArray{T, 2}; p::AbstractArray = []) where T <: Number
    res = zeros(eltype(x), length(b.basis), size(x)[2])
    @inbounds for i in 1:size(x)[2]
        res[:, i] .= b.f_(x[:, i], p)
    end
    return res
end

Base.size(b::Basis) = size(b.basis)
ModelingToolkit.parameters(b::Basis) = b.parameter
variables(b::Basis) = b.variables
parameter(b::Basis) = b.parameter

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

function Base.unique(b₀::AbstractVector{Operation})
    b = simplify_constants.(b₀)
    N = length(b)
    returns = Vector{Bool}()
    for i ∈ 1:N
        push!(returns, any([isequal(b[i], b[j]) for j in i+1:N]))
    end
    returns = [!r for r in returns]
    return b[returns]
end

function fix_single_vars_in_basis!(basis,variables)
    for (ind, el) in enumerate(basis)
        for (ind_var, var) in enumerate(variables)
            if isequal(el,var)
                basis[ind] = 1var
            end
        end
    end
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
