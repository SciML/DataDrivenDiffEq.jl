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
                    fs::Tuple; skip = false, eltype::Type{T} = Float32, parameter_mask = zeros(Bool, in_dimension), kwargs...) where {T}
    n_inputs = in_dimension
    
    input_functions = Any[]

    valid_idxs = zeros(Bool, length(fs))
    layers = map(1:(n_layers + 1)) do i
        valid_idxs .= true
        # Filter the functions by their input dimension
        valid_idxs .= (arities .<= n_inputs)
        if i <= n_layers
            layer = FunctionLayer(n_inputs, arities[valid_idxs], fs[valid_idxs];
            skip = skip, id_offset = i, input_functions = input_functions, parameter_mask = parameter_mask, kwargs...)
            push!(input_functions, fs[valid_idxs]...)
        else
            layer = FunctionLayer(n_inputs, Tuple(1 for i in 1:out_dimension),
                                  Tuple(identity for i in 1:out_dimension); skip = false,
                                  id_offset = n_layers+1,
                                  kwargs...)
            push!(input_functions, fs[valid_idxs]...)
        end

        if skip
            n_inputs = n_inputs + sum(valid_idxs)
        else
            n_inputs = sum(valid_idxs)
            empty!(input_functions)
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

function get_loglikelihood(c::LayeredDAG, ps, st)
    _get_layer_loglikelihood(c.layers, ps, st)
end

function get_loglikelihood(c::LayeredDAG, ps, st, paths::Vector{<:AbstractPathState})
    lls = get_loglikelihood(c, ps, st)
    map(paths) do path
        nodes = get_nodes(path)
        sum(map(nodes) do (i,j)
            i > 0 && return lls[i][j]
            return 0f0
        end)
    end
end
