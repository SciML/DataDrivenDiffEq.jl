
"""
$(TYPEDEF)

A container for multiple [`DecisionNodes`](@ref).

# Fields
$(FIELDS)
"""
struct DecisionLayer{skip, T, output_dimension} <: Lux.AbstractExplicitContainerLayer{(:layers,)}
    "A container for the decision nodes"
    layers::T
end

function DecisionLayer(in_dimension::Int, arities::Tuple, fs::Tuple; skip = false, kwargs...)
    layers = map(eachindex(arities)) do i 
        DecisionNode(in_dimension, arities[i], fs[i]; kwargs...)
    end
    output_dimension = length(arities)
    output_dimension += skip ? in_dimension : 0
    names = Symbol.(fs)
    layers = NamedTuple{names}(layers)
    return DecisionLayer{skip, typeof(layers), output_dimension}(layers)
end

function (r::DecisionLayer)(x, ps, st)
    _apply_layer(r.layers, x, ps, st)
end

function (r::DecisionLayer{true})(x, ps, st)
    y, st = _apply_layer(r.layers, x, ps, st)
    vcat(y, x), st
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


function update_state(r::DecisionLayer, ps, st)
    _update_state(r.layers, ps, st)
end

@generated function _update_state(layers::NamedTuple{fields}, ps,
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

Distributions.logpdf(r::DecisionLayer, ps, st) = __logpdf(r.layers, ps, st)
Distributions.pdf(r::DecisionLayer, ps, st) = __pdf(r.layers, ps, st)

@generated function __logpdf(layers::NamedTuple{fields}, ps,
    st::NamedTuple{fields}) where {fields}
    N = length(fields)
    lpdf = gensym()
    calls = [:($(lpdf) += logpdf(layers.$(fields[i]),
                                        ps.$(fields[i]),
                                        st.$(fields[i])))
    for i in 1:N]
    pushfirst!(calls, :($(lpdf) = 0))
    return Expr(:block, calls...)
end

@generated function __pdf(layers::NamedTuple{fields}, ps,
    st::NamedTuple{fields}) where {fields}
    N = length(fields)
    lpdf = gensym()
    calls = [:($(lpdf) *= pdf(layers.$(fields[i]),
                                        ps.$(fields[i]),
                                        st.$(fields[i])))
    for i in 1:N]
    pushfirst!(calls, :($(lpdf) = 1))
    return Expr(:block, calls...)
end
