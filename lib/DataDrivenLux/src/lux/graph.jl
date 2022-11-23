"""
$(TYPEDEF)

A container for a layered directed acyclic graph consisting of
different [`DecisionLayer`](@ref)s.

# Fields
$(FIELDS)
"""
struct LayeredDAG{T} <: Lux.AbstractExplicitContainerLayer{(:layers,)}
    layers::T
end

function LayeredDAG(in_dimension::Int, out_dimension::Int, n_layers::Int, arities::Tuple,
                    fs::Tuple; skip = false, eltype::Type{T} = Float32, kwargs...) where T
    n_inputs = in_dimension
    id_offset = in_dimension
    valid_idxs = zeros(Bool, length(fs))
    layers = map(1:(n_layers + 1)) do i
        valid_idxs .= true
        # Filter the functions by their input dimension
        valid_idxs .= (arities .<= n_inputs)
        
        if i <= n_layers
            layer = DecisionLayer(n_inputs, arities[valid_idxs], fs[valid_idxs];
                                  skip = skip, id_offset = i, kwargs...)
        else
            layer = DecisionLayer(n_inputs, Tuple(1 for i in 1:out_dimension),
                                  Tuple(nothing for i in 1:out_dimension); skip = false,
                                  id_offset = id_offset,
                                  kwargs...)
        end

        if skip
            n_inputs = n_inputs + sum(valid_idxs)
        else
            n_inputs = sum(valid_idxs)
        end
        layer
    end
    # The last layer is a decision node which uses an identity
    names = [Symbol("layer_$i") for i in 1:(n_layers + 1)]
    layers = NamedTuple(zip(names, layers))
    return LayeredDAG{typeof(layers)}(layers)
end


# Given that this is basically a chain, we hijack Lux
function (c::LayeredDAG)(x, ps, st)
    return Lux.applychain(c.layers, x, ps, st)
end

Base.keys(m::LayeredDAG) = Base.keys(getfield(m, :layers))

Base.getindex(c::LayeredDAG, i::Int) = c.layers[i]
Base.getindex(c::LayeredDAG, i::Int, j::Int) = getindex(c.layers[i], j)

Base.length(c::LayeredDAG) = length(c.layers)
Base.lastindex(c::LayeredDAG) = lastindex(c.layers)
Base.firstindex(c::LayeredDAG) = firstindex(c.layers)

function update_state(c::LayeredDAG, ps, st)
    _update_layer_state(c.layers, ps, st)
end

function get_loglikelihood(c::LayeredDAG, ps, st)
    _get_layer_loglikelihood(c.layers, ps, st)
end

function get_loglikelihood(c::LayeredDAG, ps, st, node_ids::Vector{Tuple{Int, Int}})
    lls = get_loglikelihood(c, ps, st)
    sum(map(node_ids) do (i,j)
        i > 0 && return lls[i][j]
        0
    end)
end
