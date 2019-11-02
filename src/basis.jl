mutable struct Basis{O, V, P} <: abstractBasis
    basis::O
    variables::V
    parameter::P
    f_
end

function Basis(basis::AbstractVector{Operation})
    bs = unique(basis)
    vs = [b for b in [ModelingToolkit.vars(bs)...] if !b.known]
    ps = [b for b in [ModelingToolkit.vars(bs)...] if b.known]

    f_ = ModelingToolkit.build_function(bs, vs, ps, (), simplified_expr, Val{false})[1]
    return Basis(bs, vs, ps, f_)
end

function update!(b::Basis)
    b.variables = [bi for bi in [ModelingToolkit.vars(b.basis)...] if !bi.known]
    b.parameter = [bi for bi in [ModelingToolkit.vars(b.basis)...] if bi.known]

    b.f_ = ModelingToolkit.build_function(b.basis, b.variables, b.parameter, (), simplified_expr, Val{false})[1]
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

(b::Basis)(x::AbstractArray) = b.f_(x, [])
(b::Basis)(x::AbstractArray; p::AbstractArray = []) = b.f_(x, p)

Base.size(b::Basis) = size(b.basis)
ModelingToolkit.parameters(b::Basis) = b.parameter
variables(b::Basis) = b.variables
#isunique(b::Basis) = length(b.basis) == length(unique(b.basis))

function jacobian(b::Basis, variables, parameters)
    j = calculate_jacobian(b.basis, variables)
    return ModelingToolkit.build_function(expand_derivatives.(j), variables, parameters, (), simplified_expr, Val{false})[1]
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
    return Basis(b.basis[returns])
end
