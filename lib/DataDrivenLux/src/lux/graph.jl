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

function LayeredDAG(in_dimension::Int, out_dimension::Int, n_layers::Int,
        fs::Vector{Pair{Function, Int}}; kwargs...)
    LayeredDAG(in_dimension, out_dimension, n_layers, tuple(last.(fs)...),
        tuple(first.(fs)...); kwargs...)
end

function LayeredDAG(in_dimension::Int, out_dimension::Int, n_layers::Int, arities::Tuple,
        fs::Tuple; skip = false, eltype::Type{T} = Float32,
        input_functions = Any[identity for i in 1:in_dimension],
        kwargs...) where {T}
    n_inputs = in_dimension

    input_functions = copy(input_functions)

    valid_idxs = zeros(Bool, length(fs))
    layers = FunctionLayer[]

    foreach(1:n_layers) do i
        valid_idxs .= true

        valid_idxs .= (arities .<= n_inputs)

        layer = FunctionLayer(n_inputs, arities[valid_idxs], fs[valid_idxs];
            skip = skip, id_offset = i, input_functions = input_functions,
            kwargs...)

        if skip
            n_inputs = n_inputs + sum(valid_idxs)
        else
            n_inputs = sum(valid_idxs)
            empty!(input_functions)
        end

        pushfirst!(input_functions, fs[valid_idxs]...)

        push!(layers, layer)
    end
    # The last layer is a decision node which uses an identity
    push!(layers,
        FunctionLayer(n_inputs, Tuple(1 for i in 1:out_dimension),
            Tuple(identity for i in 1:out_dimension);
            skip = false, input_functions = input_functions,
            id_offset = n_layers + 1, kwargs...))

    return Lux.Chain(layers...)
end

function get_loglikelihood(c::Lux.Chain, ps, st)
    _get_layer_loglikelihood(c.layers, ps, st)
end

function get_configuration(c::Lux.Chain, ps, st)
    _get_configuration(c.layers, ps, st)
end

function get_loglikelihood(c::Lux.Chain, ps, st, paths::Vector{<:AbstractPathState})
    lls = get_loglikelihood(c, ps, st)
    sum(map(paths) do path
        nodes = get_nodes(path)
        sum(map(nodes) do (i, j)
            i > 0 && return lls[i][j]
            return 0.0f0
        end)
    end)
end
