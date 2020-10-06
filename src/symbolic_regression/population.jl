struct OperationPool{F}
    ops::AbstractArray{F}
    unary::BitArray
    weights::AbstractWeights
end

OperationPool(ops::AbstractArray{Function}) = OperationPool(ops, is_unary.(ops, Number), Weights(ones(length(ops))))
is_unary(op::OperationPool, ind) = op.unary[ind]

Base.length(op::OperationPool) = length(op.ops)
Base.size(op::OperationPool, args...) = size(op.ops, args...)

function random_operation(op::OperationPool)
    idx = sample(1:length(op), op.weights)
    return (op.ops[idx], op.unary[idx])
end


mutable struct Candidate{B, S}
    basis::B
    score::S
end

Candidate(b::Basis) = Candidate(b, fill(-Inf, length(b)))

(c::Candidate)(args...) = c.basis(args...)

variables(c::Candidate) = variables(c.basis)
ModelingToolkit.parameters(c::Candidate) = parameters(c.basis)
ModelingToolkit.independent_variable(c::Candidate) = independent_variable(c.basis)
score(c::Candidate, ind = :) = c.score[ind]


Base.length(c::Candidate) = length(c.basis)
Base.size(c::Candidate, args...) = size(c.basis, args...) 

_select_features(f, rng, n) = sample(f[rng], n, replace = false, ordered = true)
_select_features(c::Candidate, rng, n) = _select_features(c.basis, rng, n)

function _conditional_feature!(features, i, op, selection_rng, maxiter = 100)
    op_ = features[1]
    op_in_features = true
    iter = 0
    while op_in_features && (iter <= maxiter)
        f,unary = random_operation(op)
        states = unary ? _select_features(features, selection_rng, 1) : _select_features(features, selection_rng, 2)
        op_ = Operation(f, states)
        op_in_features = any(map(x->isequal(op_, x), features))
        iter += 1
    end
    features[i] = op_
    return
end

function add_features!(c::Candidate, op::OperationPool, n_features::Int64 = 1, selection_rng = nothing, insertion_rng = nothing; maxiter = 100)
    n_basis = length(c)
    features = Array{Operation}(undef, n_basis+n_features)
    features[1:n_basis] .= simplify.(c.basis.basis)
    features[n_basis+1:end] .= ModelingToolkit.Constant(0)
    selection_inds = isnothing(selection_rng) ? (1:n_basis) : (1:length(features))
    for i in n_basis+1:(n_basis+n_features)
        @views _conditional_feature!(features, i, op, selection_inds, maxiter)
    end
    if !isnothing(insertion_rng)
        @views for i in insertion_rng
            c.basis.basis[i] = features[i]
        end
    else
        for f in features[n_basis+1:end]
            push!(c.basis.basis, f)
        end
    end
    update!(c.basis)
    return
end
