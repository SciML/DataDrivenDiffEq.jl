
"""
$(TYPEDEF)

A container for multiple [`DecisionNodes`](@ref).

# Fields
$(FIELDS)
"""
struct DecisionLayer{skip, T, output_dimension} <:
       Lux.AbstractExplicitContainerLayer{(:layers,)}
    "A container for the decision nodes"
    layers::T
end

function DecisionLayer(in_dimension::Int, arities::Tuple, fs::Tuple; skip = false,
                       kwargs...)
    layers = map(eachindex(arities)) do i
        DecisionNode(in_dimension, arities[i], fs[i]; kwargs...)
    end
    output_dimension = length(arities)
    output_dimension += skip ? in_dimension : 0
    names = replace(x -> isnothing(x) ? gensym("identity") : Symbol(x), fs)
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


Base.keys(m::DecisionLayer) = Base.keys(getfield(m, :layers))

Base.getindex(c::DecisionLayer, i::Int) = c.layers[i]

Base.length(c::DecisionLayer) = length(c.layers)
Base.lastindex(c::DecisionLayer) = lastindex(c.layers)
Base.firstindex(c::DecisionLayer) = firstindex(c.layers)

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

function update_state(r::DecisionLayer, ps, st)
    _update_state(r.layers, ps, st)
end

Distributions.logpdf(r::DecisionLayer, ps, st) = __logpdf(r.layers, ps, st)
Distributions.pdf(r::DecisionLayer, ps, st) = __pdf(r.layers, ps, st)

"""
$(TYPEDEF)

A wrapper for a layered directed acyclic graph.

# Fields
$(FIELDS)
"""
struct LayeredDAG{T} <: Lux.AbstractExplicitContainerLayer{(:layers,)}
    layers::T
end

function LayeredDAG(in_dimension::Int, out_dimension::Int, n_layers::Int, arities::Tuple,
                    fs::Tuple; skip = false, kwargs...)
    n_inputs = in_dimension

    layers = map(1:(n_layers + 1)) do i
        if i <= n_layers
            layer = DecisionLayer(n_inputs, arities, fs; skip = skip, kwargs...)
        else
            layer = DecisionLayer(n_inputs, Tuple(1 for i in 1:out_dimension),
                                  Tuple(nothing for i in 1:out_dimension); skip = false,
                                  kwargs...)
        end
        if skip
            n_inputs = n_inputs + length(fs)
        else
            n_inputs = length(arities)
        end
        layer
    end

    # The last layer is a decision node which uses an identity
    names = [Symbol("layer_$i") for i in 1:(n_layers + 1)]
    layers = NamedTuple(zip(names, layers))
    return LayeredDAG{typeof(layers)}(layers)
end

function update_state(r::LayeredDAG, ps, st)
    _update_state(r.layers, ps, st)
end

Distributions.logpdf(r::LayeredDAG, ps, st) = __logpdf(r.layers, ps, st)
Distributions.pdf(r::LayeredDAG, ps, st) = __pdf(r.layers, ps, st)



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

## Collect the traversal information
function __get_input(st::NamedTuple{(:loglikelihood, :input_id, :temperature, :rng)})
    st.input_id
end

@generated function __get_input(st::NamedTuple{fields, <: Tuple}) where fields
    N = length(fields)
    outputs = [gensym() for i in 1:N]
    calls = [:($(outputs[i]) = __get_input(st.$(fields[i]))) for i in reverse(1:N)]
    push!(calls, :($(Tuple(reverse(outputs))...),))
    return Expr(:block, calls...)
end

function __get_path(path::NTuple{N, Tuple}, id::Union{Int, Nothing} = nothing) where {N, M}
    next_ = first(path)
    tails_ = Base.tail(path)
    if isnothing(id)
        return map(next_) do id
            (id, __get_path(tails_, id)...)
        end
    elseif !isempty(tails_)
        next_id = getindex(next_, id)
        next_ids = __get_path(tails_, next_id)
        return (next_id, next_ids...)
    else
        return (getindex(next_, id))
    end
end

function __get_path(path::NTuple{M, Int}, id::Int) where {M}
    (getindex(path, id))
end

function get_path(d::LayeredDAG, ps, st)
    __get_path(__get_input(st), nothing)
end

# TODO Make me faster
function StatsBase.dof(d::LayeredDAG, ps, st::NamedTuple{fields})::Int where {fields}
    paths = get_path(d, ps, st)
    dof = 0
    @inbounds foreach(1:length(d)) do i 
        dof += length(union(map(Base.Fix2(getindex, i), paths)))
    end
    dof
end