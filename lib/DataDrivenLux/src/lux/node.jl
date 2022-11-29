
"""
$(TYPEDEF)

A layer representing a decision node with a single function 
and a latent array of weights representing a probability distribution over the inputs.

# Fields
$(FIELDS)

"""
struct DecisionNode{skip, W, S, F, ID} <: Lux.AbstractExplicitLayer
    "Input dimensions of the signal"
    in_dims::Int
    "Function which should map from in_dims ↦ R"
    f::F
    "Arity of the function"
    arity::Int
    "Weight initialization"
    init_weight::W
    "Mapping to the unit simplex"
    simplex::S
end

function DecisionNode(in_dims::Int, arity::Int, f::F = identity;
                      init_weight = Lux.zeros32, skip = false,
                      simplex = Softmax(), id = 0,
                      kwargs...) where {F}
    return DecisionNode{skip, typeof(init_weight), typeof(simplex), F, id}(in_dims, f, arity,
                                                                       init_weight, simplex)
end

get_id(::DecisionNode{<:Any, <:Any, <:Any, <:Any, ID}) where ID = ID

function Lux.initialparameters(rng::AbstractRNG, l::DecisionNode)
    return (; weight = l.init_weight(rng, l.arity, l.in_dims))
end

function Lux.initialstates(rng::AbstractRNG, p::DecisionNode)
    begin
        rand(rng)
        rng_ = Lux.replicate(rng)
        # Call once

        (loglikelihood = zeros(Float32, p.arity),
         input_id = zeros(Int, p.arity),
         temperature = 1.0f0,
         rng = rng_)
    end
end

function update_state(p::DecisionNode, ps, st)
    @unpack temperature, rng, loglikelihood, input_id = st
    @unpack weight = ps

    # Transform to the unit simplex
    priors = p.simplex(rng, weight, temperature)

    foreach(1:p.arity) do i 
        dist = Categorical(priors[i, :])
        input_id[i] = rand(rng, dist)
        loglikelihood[i] = logpdf(dist, input_id[i])
    end

    (;
     loglikelihood = loglikelihood,
     input_id = input_id,
     temperature = temperature,
     rng = rng
     )
end

function (l::DecisionNode{false})(x::AbstractArray{<:Number}, ps, st::NamedTuple)
    y = _apply_node(l, x, ps, st)
    return y, st
end

function (l::DecisionNode{true})(x::AbstractArray{<:Number}, ps, st::NamedTuple)
    y = _apply_node(l, x, ps, st)
    return vcat(y, x), st
end

# Special dispatch on the path state
function (l::DecisionNode{false})(x::AbstractArray{<:AbstractPathState}, ps, st::NamedTuple)
    new_st = update_state(l, ps, st)
    @unpack input_id, loglikelihood = new_st
    update_path(l.f, sum(loglikelihood), get_id(l), x[input_id]...), new_st
end

function (l::DecisionNode{true})(x::AbstractArray{<:AbstractPathState}, ps, st::NamedTuple)
    new_st = update_state(l, ps, st)
    @unpack input_id, loglikelihood = new_st
    vcat(update_path(l.f, sum(loglikelihood), get_id(l), x[input_id]...), x), new_st
end


function _apply_node(l::DecisionNode, x::AbstractMatrix, ps, st)::AbstractMatrix
    @unpack input_id = st
    reduce(hcat, map(eachcol(x)) do xi
        _apply_node(l, xi, ps, st)
    end)
end

function _apply_node(l::DecisionNode, x::AbstractVector, ps, st)
    @unpack input_id = st
    map(l.f, x[input_id]...)
end

function _apply_node(l::DecisionNode{<:Any, <:Any, <:Any, Nothing}, x::AbstractVector, ps,
                     st)
    @unpack input_id = st
    x[input_id]
end

function _apply_node(l::DecisionNode{<:Any, <:Any, <:Any, Nothing}, x::AbstractMatrix, ps,
    st)::AbstractMatrix
    @unpack input_id = st
    x[input_id, :]
end

function set_temperature(::DecisionNode, temperature, ps, st)
    merge(st, (; temperature = temperature))
end

get_temperature(::DecisionNode, ps, st) = st.temperature

function get_loglikelihood(d::DecisionNode, ps, st)
    logll = logsoftmax(ps.weight ./ st.temperature , dims = 2)
    return sum(logll[st.input_id])
end 

get_inputs(::DecisionNode, ps, st) = st.input_id
