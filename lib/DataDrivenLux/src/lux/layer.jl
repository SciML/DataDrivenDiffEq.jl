"""
$(TYPEDEF)

A container for multiple [`FunctionNode`](@ref)s.
It accumulates all outputs of the nodes.

# Fields

$(FIELDS)

# Arguments

- `in_dimension::Int`: number of available input signals.
- `arities::Tuple`: arity for each function in `fs`.
- `fs::Tuple`: functions used to construct the nodes.

# Keywords

- `skip::Bool`: include a skip connection around the layer.
- `id_offset::Int`: starting layer identifier for path bookkeeping.
- `input_functions`: optional input functions used by each node.

# Returns

Return a Lux wrapper layer whose output contains the values of all nodes.
"""
@concrete struct FunctionLayer <: AbstractLuxWrapperLayer{:nodes}
    nodes
    skip
end

function FunctionLayer(
        in_dimension::Int, arities::Tuple, fs::Tuple; skip = false, id_offset = 1,
        input_functions = Any[identity for i in 1:in_dimension], kwargs...
    )
    nodes = map(eachindex(arities)) do i
        # We check if we have an inverse here
        return FunctionNode(
            fs[i], arities[i], in_dimension, (id_offset, i);
            input_functions, kwargs...
        )
    end
    inner_model = Lux.Chain(Lux.BranchLayer(nodes...), Lux.WrappedFunction(splat(vcat)))
    return FunctionLayer(
        skip ? Lux.Parallel(vcat, inner_model, Lux.NoOpLayer()) : inner_model, skip
    )
end

function get_loglikelihood(r::FunctionLayer, ps, st)
    if r.skip
        return _get_layer_loglikelihood(
            r.nodes.layers[1].layers[1].layers, ps.layer_1.layer_1, st.layer_1.layer_1
        )
    else
        return _get_layer_loglikelihood(r.nodes.layers[1].layers, ps.layer_1, st.layer_1)
    end
end

function get_configuration(r::FunctionLayer, ps, st)
    if r.skip
        return _get_configuration(
            r.nodes.layers[1].layers[1].layers, ps.layer_1.layer_1, st.layer_1.layer_1
        )
    else
        return _get_configuration(r.nodes.layers[1].layers, ps.layer_1, st.layer_1)
    end
end

@generated function _get_layer_loglikelihood(
        layers::NamedTuple{fields}, ps, st::NamedTuple{fields}
    ) where {fields}
    N = length(fields)
    st_symbols = [gensym() for _ in 1:N]
    calls = [
        :(
            $(st_symbols[i]) = get_loglikelihood(
                layers.$(fields[i]), ps.$(fields[i]), st.$(fields[i])
            )
        ) for i in 1:N
    ]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    return Expr(:block, calls...)
end

@generated function _get_configuration(
        layers::NamedTuple{fields}, ps, st::NamedTuple{fields}
    ) where {fields}
    N = length(fields)
    st_symbols = [gensym() for _ in 1:N]
    calls = [
        :(
            $(st_symbols[i]) = get_configuration(
                layers.$(fields[i]), ps.$(fields[i]), st.$(fields[i])
            )
        ) for i in 1:N
    ]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    return Expr(:block, calls...)
end
