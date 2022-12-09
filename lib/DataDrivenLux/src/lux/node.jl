
"""
$(TYPEDEF)

A layer representing a decision node with a single function 
and a latent array of weights representing a probability distribution over the inputs.

# Fields
$(FIELDS)

"""
struct FunctionNode{skip, ID, F, W, S} <: Lux.AbstractExplicitLayer
    "Function which should map from in_dims ↦ R"
    f::F
    "Arity of the function"
    arity::Int
    "Input dimensions of the signal"
    in_dims::Int
    "Mapping to the unit simplex"
    simplex::S
    "Masking of the input values"
    input_mask::Vector{Bool}
    "Weight initialization"
    init_weight::W
end

function FunctionNode(f::F, arity::Int, input_dimension::Int, id::Union{Int, NTuple{M, Int} where M}; 
        init_weight = Lux.zeros32, skip = false, simplex = Softmax(), 
        input_mask = ones(Bool, input_dimension)
    ) where {F}
    @assert all(sum(input_mask) .>= 1) "Input masks should enable at least one choice."
    return FunctionNode{skip, id, F, typeof(init_weight), typeof(simplex)}(
        f, arity, input_dimension, simplex, input_mask, init_weight
    )
end

get_id(::FunctionNode{<:Any, id}) where id = id

function Lux.initialparameters(rng::AbstractRNG, l::FunctionNode)
    weights = tuple(collect(l.init_weight(rng, sum(l.input_mask)) for i in 1:l.arity)...)
    return (; weights = weights)    
end

function Lux.initialstates(rng::AbstractRNG, p::FunctionNode)
    begin
        rand(rng)
        rng_ = Lux.replicate(rng)
        # Call once
        (;
         active_inputs = zeros(Int, p.arity),
         temperature = 1.0f0,
         priors = tuple(collect(p.init_weight(rng_, p.in_dims) for i in 1:p.arity)...),
         rng = rng_)
    end
end


function update_state(p::FunctionNode, ps, st)
    @unpack temperature, rng, active_inputs, priors = st
    @unpack weights = ps

    foreach(1:(p.arity)) do i
        priors[i] .= p.simplex(rng, weights[i], temperature)
        active_inputs[i] = findfirst(rand(rng) .< cumsum(priors[i]))
    end

    (;
     active_inputs = active_inputs,
     temperature = temperature,
     priors = priors,
     rng = rng)
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
    ntuple(i->x[l.input_mask][st.active_inputs[i]], l.arity)
end

@views function _apply_node(l::FunctionNode, x::AbstractVector, ps, st::NamedTuple{fieldnames}) where {fieldnames}
    @unpack active_inputs = st
    @unpack input_mask = l
    l.f(get_masked_inputs(l, x, ps, st)...)
end

function set_temperature(::FunctionNode, temperature, ps, st)
    merge(st, (; temperature = temperature))
end

get_temperature(::FunctionNode, ps, st) = st.temperature

function get_loglikelihood(d::FunctionNode, ps, st)
    #logll = logsoftmax(ps.weight ./ st.temperature, dims = 2)
    sum(map(enumerate(ps.weights)) do (i, weight)
        logsoftmax(weight ./ st.temperature)[st.active_inputs[i]]
    end)
end

get_inputs(::FunctionNode, ps, st) = st.active_inputs

# Special dispatch on the path state
function (l::FunctionNode{false})(x::AbstractVector{<:AbstractPathState}, ps, st::NamedTuple)
    new_st = update_state(l, ps, st)
    update_path(l.f, get_id(l), get_masked_inputs(l, x, ps, new_st)...), new_st
end

function (l::FunctionNode{true})(x::AbstractVector{<:AbstractPathState}, ps, st::NamedTuple)
    new_st = update_state(l, ps, st)
    vcat(update_path(l.f, get_id(l), get_masked_inputs(l, x, ps, new_st)...), x), new_st
end

