
"""
$(TYPEDEF)

A layer representing a decision node with a single function 
and a latent array of weights representing a probability distribution over the inputs.

# Fields
$(FIELDS)

"""
@concrete struct FunctionNode{skip, ID} <: AbstractLuxLayer
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
    map(xi -> mask_inverse(f, arity, xi), in_f)
end
mask_inverse(f::F, arity::Int, val::Bool) where {F <: Function} = arity == 1 ? val : true
function mask_inverse(f::F, arity::Int, g::G) where {F <: Function, G <: Function}
    InverseFunctions.inverse(f) != g
end
mask_inverse(::typeof(+), arity::Int, in_f::AbstractVector) = ones(Bool, length(in_f))
mask_inverse(::typeof(-), arity::Int, in_f::AbstractVector) = ones(Bool, length(in_f))
function mask_inverse(::typeof(identity), arity::Int, in_f::AbstractVector)
    ones(Bool, length(in_f))
end

function FunctionNode(f::F, arity::Int, input_dimension::Int,
        id::Union{Int, NTuple{M, Int} where M};
        skip = false, simplex = Softmax(),
        input_functions = [identity for i in 1:input_dimension],
        kwargs...) where {F}
    input_mask = mask_inverse(f, arity, input_functions)

    @assert sum(input_mask)>=1 "Input masks should enable at least one choice."
    @assert length(input_mask)==input_dimension "Input dimension should be sized equally to input_mask"

    return FunctionNode{skip, id, F, typeof(simplex)}(f, arity,
        input_dimension,
        simplex,
        input_mask)
end

get_id(::FunctionNode{<:Any, id}) where {id} = id

function LuxCore.initialparameters(rng::AbstractRNG, l::FunctionNode)
    return (; weights = init_weights(l.simplex, rng, sum(l.input_mask), l.arity))
end

function LuxCore.initialstates(rng::AbstractRNG, p::FunctionNode)
    rand(rng)
    rng_ = LuxCore.replicate(rng)
    # Call once
    return (;
        priors = init_weights(p.simplex, rng, sum(p.input_mask), p.arity),
        active_inputs = zeros(Int, p.arity),
        temperature = 1.0f0,
        rng = rng_)
end

function update_state(p::FunctionNode, ps, st)
    @unpack temperature, rng, active_inputs, priors = st
    @unpack weights = ps

    foreach(enumerate(eachcol(weights))) do (i, weight)
        @views p.simplex(rng, priors[:, i], weight, temperature)
        active_inputs[i] = findfirst(rand(rng) .<= cumsum(priors[:, i]))
    end

    return (; priors, active_inputs, temperature, rng)
end

function (l::FunctionNode{false})(x::AbstractArray{<:Number}, ps, st::NamedTuple)
    y = _apply_node(l, x, ps, st)
    return y, st
end

function (l::FunctionNode{true})(x::AbstractArray{<:Number}, ps, st::NamedTuple)
    y = _apply_node(l, x, ps, st)
    return vcat(y, x), st
end

@views function _apply_node(l::FunctionNode, x::AbstractMatrix, ps, st)::AbstractMatrix
    reduce(hcat, map(eachcol(x)) do xi
        _apply_node(l, xi, ps, st)
    end)
end

function get_masked_inputs(l::FunctionNode, x::AbstractVector, ps, st::NamedTuple)
    @unpack active_inputs = st
    @unpack input_mask = l
    ntuple(i -> x[input_mask][active_inputs[i]], l.arity)
end

@views function _apply_node(l::FunctionNode, x::AbstractVector, ps,
        st::NamedTuple{fieldnames}) where {fieldnames}
    l.f(get_masked_inputs(l, x, ps, st)...)
end

function set_temperature(::FunctionNode, temperature, ps, st)
    merge(st, (; temperature = temperature))
end

get_temperature(::FunctionNode, ps, st) = st.temperature

function get_loglikelihood(d::FunctionNode, ps, st)
    @unpack weights = ps
    sum(map(enumerate(eachcol(weights))) do (i, weight)
        logsoftmax(weight ./ st.temperature)[st.active_inputs[i]]
    end)
end

get_inputs(::FunctionNode, ps, st) = st.active_inputs

function get_configuration(::FunctionNode, ps, st)
    @unpack weights = ps
    @unpack active_inputs = st
    config = similar(weights)
    xzero = zero(eltype(config))
    xone = one(eltype(config))
    foreach(enumerate(eachcol(config))) do (i, config_)
        config_ .= xzero
        config_[active_inputs[i]] = xone
    end
    (; weights = config)
end

# Special dispatch on the path state
function (l::FunctionNode{false})(x::AbstractVector{<:AbstractPathState}, ps,
        st::NamedTuple)
    new_st = update_state(l, ps, st)
    update_path(l.f, get_id(l), get_masked_inputs(l, x, ps, new_st)...), new_st
end

function (l::FunctionNode{true})(x::AbstractVector{<:AbstractPathState}, ps, st::NamedTuple)
    new_st = update_state(l, ps, st)
    vcat(update_path(l.f, get_id(l), get_masked_inputs(l, x, ps, new_st)...), x), new_st
end
