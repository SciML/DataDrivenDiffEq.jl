import Base.==


mutable struct Basis{O, V, P} <: abstractBasis
    basis::O
    variables::V
    parameter::P
    f_
end

Base.show(io::IO, x::Basis) = print(io, "$(length(x.basis)) dimensional basis in ", "$(String.([v.op.name for v in x.variables]))")

@inline function Base.print(io::IO, x::Basis)
    show(io, x)
    println()
    if length(x.variables) == length(x.basis)
        for (i, bi) in enumerate(x.basis)
            println("d$(x.variables[i]) = $bi")
        end
    else
        for (i, bi) in enumerate(x.basis)
            println("f_$i = $bi")
        end
    end
end

is_independent(o::Operation) = isempty(o.args)

function Basis(basis::AbstractVector{Operation}, variables::AbstractVector{Operation};  parameters =  [])
    @assert all(is_independent.(variables)) "Please provide independent variables for base."

    bs = unique(basis)
    fix_single_vars_in_basis!(bs, variables)

    vs = [ModelingToolkit.Variable(Symbol(i)) for i in variables]
    ps = [ModelingToolkit.Variable(Symbol(i)) for i in parameters]

    f_ = ModelingToolkit.build_function(bs, vs, ps, (), simplified_expr, Val{false})[1]
    return Basis(bs, variables, parameters, f_)
end

function update!(basis::Basis)
    vs = [ModelingToolkit.Variable(Symbol(i)) for i in variables(basis)]
    ps = [ModelingToolkit.Variable(Symbol(i)) for i in parameters(basis)]

    basis.f_ = ModelingToolkit.build_function(basis.basis, vs, ps, (), simplified_expr, Val{false})[1]
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
    basis_a.parameter = unique(vcat(basis_a.parameter, basis_b.parameter))
    update!(basis_a)
    return
end

Base.getindex(b::Basis, idx::Int64) = b.basis[idx]
Base.getindex(b::Basis, ids::UnitRange{Int64}) = b.basis[ids]
Base.getindex(b::Basis, ::Colon) = b.basis
Base.firstindex(b::Basis) = firstindex(b.basis)
Base.lastindex(b::Basis) = lastindex(b.basis)
Base.iterate(b::Basis) = iterate(b.basis)
Base.iterate(b::Basis, id::Int64) = iterate(b.basis, id)

function (==)(x::Basis, y::Basis)
    n = zeros(Bool, length(x.basis))
    @inbounds for (i, xi) in enumerate(x)
        n[i] = any(isequal.(xi, y.basis))
    end
    return all(n)
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

(b::Basis)(x::AbstractArray{T, 1}; p::AbstractArray = []) where T <: Number = b.f_(x, isempty(p) ? parameters(b) : p)



function (b::Basis)(x::AbstractArray{T, 2}; p::AbstractArray = []) where T <: Number
    if (isempty(p) || eltype(p) <: Expression) && !isempty(parameters(b))
        pi = isempty(p) ? parameters(b) : p
        res = zeros(eltype(pi), length(b), size(x)[2])
    else
        pi = p
        res = zeros(eltype(x), length(b), size(x)[2])
    end
    @inbounds for i in 1:size(x)[2]
        res[:, i] .= b.f_(x[:, i], pi)
    end
    return res
end

Base.size(b::Basis) = size(b.basis)
Base.length(b::Basis) = length(b.basis)
ModelingToolkit.parameters(b::Basis) = b.parameter
variables(b::Basis) = b.variables

function jacobian(basis::Basis)

    vs = [ModelingToolkit.Variable(Symbol(i)) for i in variables(basis)]
    ps = [ModelingToolkit.Variable(Symbol(i)) for i in parameters(basis)]

    j = calculate_jacobian(basis.basis, variables(basis))
    return ModelingToolkit.build_function(expand_derivatives.(j), vs, ps, (), simplified_expr, Val{false})[1]
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
    b = simplify_constants.(b₀)
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
    @assert length(b) == length(variables(b))
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

function ModelingToolkit.ODESystem(b::Basis, independent_variable::Operation)
    @assert length(b) == length(variables(b))-1
    @derivatives D'~independent_variable

    vars = [vi for vi in variables(b) if ! isequal(vi, independent_variable)]

    vs = similar(vars)
    dvs = similar(vars)


    for (i, vi) in enumerate(vars)
        vs[i] = ModelingToolkit.Operation(vi.op, [independent_variable])
        dvs[i] = D(vs[i])
    end
    #return vs, dvs
    eqs = dvs .~ b([vs..., independent_variable], p = b.parameter)
    return ODESystem(eqs, independent_variable, vs, b.parameter)
end
