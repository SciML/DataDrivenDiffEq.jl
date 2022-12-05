"""
$(TYPEDEF)

A container for multiple [`DecisionNodes`](@ref). 
It accumulates all outputs of the nodes.

# Fields
$(FIELDS)
"""
struct DecisionLayer{skip, T, output_dimension} <:
       Lux.AbstractExplicitContainerLayer{(:nodes,)}
    nodes::T
end

mask_inverse(f::F, in_f) where F <: Function =  [InverseFunctions.inverse(f) != f_ for f_ in in_f]
mask_inverse(::typeof(+), in_f) = ones(Bool, length(in_f))
mask_inverse(::typeof(-), in_f) = ones(Bool, length(in_f))
mask_inverse(::Nothing, in_f) = ones(Bool, length(in_f))

function mask_parameters(arity, parameter_mask)
    arity <= 1 && return .! parameter_mask
    return ones(Bool, length(parameter_mask))
end

function DecisionLayer(in_dimension::Int, arities::Tuple, fs::Tuple; skip = false,
                       id_offset = 1, input_functions = (), parameter_mask = zeros(Bool, in_dimension),
                       kwargs...)

    nodes = map(eachindex(arities)) do i
        # We check if we have an inverse here
        local_input_mask = vcat(mask_inverse(fs[i], input_functions), mask_parameters(arities[i], parameter_mask))
        DecisionNode(in_dimension, arities[i], fs[i]; id = (id_offset, 1), input_mask = local_input_mask, kwargs...)
    end

    output_dimension = length(arities)
    output_dimension += skip ? in_dimension : 0

    names = map(gensym ∘ string, fs)
    nodes = NamedTuple{names}(nodes)
    return DecisionLayer{skip, typeof(nodes), output_dimension}(nodes)
end

function (r::DecisionLayer)(x, ps, st)
    _apply_layer(r.nodes, x, ps, st)
end

function (r::DecisionLayer{true})(x, ps, st)
    y, st = _apply_layer(r.nodes, x, ps, st)
    vcat(y, x), st
end

Base.keys(m::DecisionLayer) = Base.keys(getfield(m, :nodes))

Base.getindex(c::DecisionLayer, i::Int) = c.nodes[i]

Base.length(c::DecisionLayer) = length(c.nodes)
Base.lastindex(c::DecisionLayer) = lastindex(c.nodes)
Base.firstindex(c::DecisionLayer) = firstindex(c.nodes)

function update_state(r::DecisionLayer, ps, st)
    _update_layer_state(r.nodes, ps, st)
end

function get_loglikelihood(r::DecisionLayer, ps, st)
    _get_layer_loglikelihood(r.nodes, ps, st)
end

function get_inputs(r::DecisionLayer, ps, st)
    _get_layer_inputs(r.nodes, ps, st)
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
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    push!(calls, :(return vcat($(y_symbols...)), st))
    return Expr(:block, calls...)
end

@generated function _update_layer_state(layers::NamedTuple{fields}, ps,
                                        st::NamedTuple{fields}) where {fields}
    N = length(fields)
    st_symbols = [gensym() for _ in 1:N]
    calls = [:($(st_symbols[i]) = update_state(layers.$(fields[i]),
                                               ps.$(fields[i]),
                                               st.$(fields[i])))
             for i in 1:N]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    return Expr(:block, calls...)
end

@generated function _get_layer_inputs(layers::NamedTuple{fields}, ps,
                                      st::NamedTuple{fields}) where {fields}
    N = length(fields)
    st_symbols = [gensym() for _ in 1:N]
    calls = [:($(st_symbols[i]) = get_inputs(layers.$(fields[i]),
                                             ps.$(fields[i]),
                                             st.$(fields[i])))
             for i in 1:N]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    return Expr(:block, calls...)
end
