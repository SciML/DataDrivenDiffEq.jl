"""
$(TYPEDEF)

A container for multiple [`DecisionNodes`](@ref). 
It accumulates all outputs of the nodes.

# Fields
$(FIELDS)
"""
struct FunctionLayer{skip, T, output_dimension} <: AbstractLuxWrapperLayer{:nodes}
    nodes::T
end

function FunctionLayer(in_dimension::Int, arities::Tuple, fs::Tuple; skip = false,
        id_offset = 1,
        input_functions = Any[identity for i in 1:in_dimension],
        kwargs...)
    nodes = map(eachindex(arities)) do i
        # We check if we have an inverse here
        FunctionNode(fs[i], arities[i], in_dimension, (id_offset, i);
            input_functions = input_functions, kwargs...)
    end

    output_dimension = length(arities)
    output_dimension += skip ? in_dimension : 0

    names = map(gensym ∘ string, fs)
    nodes = NamedTuple{names}(nodes)
    return FunctionLayer{skip, typeof(nodes), output_dimension}(nodes)
end

function (r::FunctionLayer)(x, ps, st)
    _apply_layer(r.nodes, x, ps, st)
end

function (r::FunctionLayer{true})(x, ps, st)
    y, st = _apply_layer(r.nodes, x, ps, st)
    vcat(y, x), st
end

Base.keys(m::FunctionLayer) = Base.keys(getfield(m, :nodes))

Base.getindex(c::FunctionLayer, i::Int) = c.nodes[i]

Base.length(c::FunctionLayer) = length(c.nodes)
Base.lastindex(c::FunctionLayer) = lastindex(c.nodes)
Base.firstindex(c::FunctionLayer) = firstindex(c.nodes)

function get_loglikelihood(r::FunctionLayer, ps, st)
    _get_layer_loglikelihood(r.nodes, ps, st)
end

function get_configuration(r::FunctionLayer, ps, st)
    _get_configuration(r.nodes, ps, st)
end

@generated function _get_layer_loglikelihood(layers::NamedTuple{fields}, ps,
        st::NamedTuple{fields}) where {fields}
    N = length(fields)
    st_symbols = [gensym() for _ in 1:N]
    calls = [:($(st_symbols[i]) = get_loglikelihood(layers.$(fields[i]),
                 ps.$(fields[i]),
                 st.$(fields[i])))
             for i in 1:N]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    return Expr(:block, calls...)
end

@generated function _get_configuration(layers::NamedTuple{fields}, ps,
        st::NamedTuple{fields}) where {fields}
    N = length(fields)
    st_symbols = [gensym() for _ in 1:N]
    calls = [:($(st_symbols[i]) = get_configuration(layers.$(fields[i]),
                 ps.$(fields[i]),
                 st.$(fields[i])))
             for i in 1:N]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    return Expr(:block, calls...)
end

@generated function _apply_layer(layers::NamedTuple{fields}, x, ps,
        st::NamedTuple{fields}) where {fields}
    N = length(fields)
    y_symbols = vcat([gensym() for _ in 1:N])
    st_symbols = [gensym() for _ in 1:N]
    calls = [:(($(y_symbols[i]), $(st_symbols[i])) = Lux.apply(layers.$(fields[i]),
                 x,
                 ps.$(fields[i]),
                 st.$(fields[i])))
             for i in 1:N]
    push!(calls, :(st = NamedTuple{$fields}(($(Tuple(st_symbols)...),))))
    push!(calls, :(return vcat($(y_symbols...)), st))
    return Expr(:block, calls...)
end
