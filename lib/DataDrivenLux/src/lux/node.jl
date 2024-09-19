
"""
$(TYPEDEF)

A layer representing a decision node with a single function 
and a latent array of weights representing a probability distribution over the inputs.

# Fields
$(FIELDS)

"""
@concrete struct FunctionNode <: AbstractLuxWrapperLayer{:node}
    node
end

@concrete struct InternalFunctionNode{ID} <: AbstractLuxLayer
    "Function which should map from in_dims ↦ R"
    f
    "Arity of the function"
    arity::Int
    "Input dimensions of the signal"
    in_dims::Int
    "Mapping to the unit simplex"
    simplex
    "Masking of the input values"
    input_mask::Vector{Bool}
end

function mask_inverse(f::F, arity::Int, in_f::AbstractVector) where {F <: Function}
    return map(xi -> mask_inverse(f, arity, xi), in_f)
end
mask_inverse(::Function, arity::Int, val::Bool) = ifelse(arity == 1, val, true)
function mask_inverse(f::F, ::Int, g::G) where {F <: Function, G <: Function}
    return InverseFunctions.inverse(f) != g
end
mask_inverse(::typeof(+), arity::Int, in_f::AbstractVector) = ones(Bool, length(in_f))
mask_inverse(::typeof(-), arity::Int, in_f::AbstractVector) = ones(Bool, length(in_f))
function mask_inverse(::typeof(identity), arity::Int, in_f::AbstractVector)
    return ones(Bool, length(in_f))
end

function FunctionNode(f::F, arity::Int, input_dimension::Int,
        id::Union{Int, NTuple{<:Any, Int}}; skip = false, simplex = Softmax(),
        input_functions = [identity for i in 1:input_dimension], kwargs...) where {F}
    input_mask = mask_inverse(f, arity, input_functions)

    @assert sum(input_mask)>=1 "Input masks should enable at least one choice."
    @assert length(input_mask)==input_dimension "Input dimension should be sized equally \
                                                 to input_mask"

    internal_node = InternalFunctionNode{id}(f, arity, input_dimension, simplex, input_mask)
    node = skip ? Lux.Parallel(vcat, internal_node, Lux, NoOpLayer()) : internal_node
    return FunctionNode(node)
end

get_id(::InternalFunctionNode{id}) where {id} = id

function LuxCore.initialparameters(rng::AbstractRNG, l::InternalFunctionNode)
    return (; weights = init_weights(l.simplex, rng, sum(l.input_mask), l.arity))
end

function LuxCore.initialstates(rng::AbstractRNG, p::InternalFunctionNode)
    rand(rng)
    rng_ = LuxCore.replicate(rng)
    return (; priors = init_weights(p.simplex, rng, sum(p.input_mask), p.arity),
        active_inputs = zeros(Int, p.arity), temperature = 1.0f0, rng = rng_)
end

@views function update_state(p::InternalFunctionNode, ps, st)
    (; temperature, rng, active_inputs, priors) = st

    foreach(enumerate(eachcol(ps.weights))) do (i, weight)
        p.simplex(rng, priors[:, i], weight, temperature)
        return active_inputs[i] = findfirst(rand(rng) .<= cumsum(priors[:, i]))
    end

    return (; priors, active_inputs, temperature, rng)
end

function (l::InternalFunctionNode)(x::AbstractMatrix, ps, st)
    return mapreduce(hcat, eachcol(x)) do xi
        return LuxCore.apply(l, xi, ps, st)
    end
end

function (l::InternalFunctionNode)(x::AbstractVector, ps, st)
    return l.f(get_masked_inputs(l, x, ps, st)...)
end

function (l::InternalFunctionNode)(x::AbstractVector{<:AbstractPathState}, ps, st)
    new_st = update_state(l, ps, st)
    return update_path(l.f, get_id(l), get_masked_inputs(l, x, ps, new_st)...), new_st
end

function get_masked_inputs(l::InternalFunctionNode, x::AbstractVector, _, st::NamedTuple)
    return ntuple(i -> x[l.input_mask][st.active_inputs[i]], l.arity)
end

get_temperature(::FunctionNode, ps, st) = st.temperature

function get_loglikelihood(::FunctionNode, ps, st)
    return sum(map(enumerate(eachcol(ps.weights))) do (i, weight)
        return logsoftmax(weight ./ st.temperature)[st.active_inputs[i]]
    end)
end

get_inputs(::FunctionNode, ps, st) = st.active_inputs

function get_configuration(::FunctionNode, ps, st)
    config = similar(ps.weights)
    foreach(enumerate(eachcol(config))) do (i, config_)
        config_ .= false
        return config_[st.active_inputs[i]] = true
    end
    return (; weights = config)
end
