ModelingToolkit.RuntimeGeneratedFunctions.init(@__MODULE__)

mutable struct Basis{O, V, P} <: abstractBasis
    basis::O
    variables::V
    parameter::P
    f_
end

is_independent(o::Term) = isempty(o.args)
is_independent(sym::Sym) = true

Base.print(io::IO, x::Basis) = show(io, x)
Base.show(io::IO, x::Basis) = print(io, "$(length(x.basis)) dimensional basis in ", "$(String.([v.op.name for v in x.variables]))")

function Basis(basis::AbstractVector, variables::AbstractVector;  parameters =  [])
    basis = Any[value.(basis)...]
    variables = value.(variables)
    @assert all(is_independent.(variables)) "Please provide independent variables for base."

    bs = unique(basis)
    vs, ps = get_vars_params(bs)
    f_ = ModelingToolkit.build_function(bs, vs, ps, conv=toexpr∘simplify)[1]

    return Basis(bs, variables, parameters, ModelingToolkit.@RuntimeGeneratedFunction(f_))
end

function get_vars_params(basis)
    vss = collect(ModelingToolkit.vars(basis))
    vs = filter(!isparameter, vss)
    ps = filter(isparameter, vss)
    sort!(vs, lt = <ₑ), sort!(ps, lt = <ₑ)
end

function update!(b::Basis)
    vs, ps = get_vars_params(b.basis)
    f_ = ModelingToolkit.build_function(b.basis, vs, ps, conv=toexpr∘simplify)[1]
    b.f_ = ModelingToolkit.@RuntimeGeneratedFunction(f_)
    return
end

function Base.push!(b::Basis, op)
    push!(b.basis, value(op))
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
    vs, ps = get_vars_params(b.basis)
    j = ModelingToolkit.jacobian(b.basis, variables(b))
    ModelingToolkit.@RuntimeGeneratedFunction(
        ModelingToolkit.build_function(expand_derivatives.(j), vs, ps, conv=toexpr∘simplify)[1])
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

make_sym_a_func(s::Sym, t) = Sym{FnType{Tuple, Real}}(nameof(s))(t)

function ModelingToolkit.ODESystem(b::Basis)
    # Define the time
    @parameters t
    @derivatives D'~t

    vs = similar(b.variables, Any)
    dvs = similar(b.variables, Any)
    for (i, vi) in enumerate(b.variables)
        vs[i] = make_sym_a_func(vi, t)
        dvs[i] = D(vs[i])
    end
    eqs = dvs .~ b(vs, p = b.parameter)
    return ODESystem(eqs)
end
