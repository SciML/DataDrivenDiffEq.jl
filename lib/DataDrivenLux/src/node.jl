
"""
using Base: ident_cmp
$(TYPEDEF)

A layer representing a decision node. 

# Fields
$(FIELDS)

"""
struct DecisionNode{skip, W, S, F} <: Lux.AbstractExplicitLayer
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
                      simplex = GumbelSoftmax(),
                      kwargs...) where {F}
    return DecisionNode{skip, typeof(init_weight), typeof(simplex), F}(in_dims, f, arity,
                                                                       init_weight, simplex)
end

function Lux.initialparameters(rng::AbstractRNG, l::DecisionNode)
    return (; weight = l.init_weight(rng, l.arity, l.in_dims))
end

function Lux.initialstates(rng::AbstractRNG, p::DecisionNode)
    begin
        rng_ = copy(rng)
        # Call once
        rand(rng, 1)

        (loglikelihood = Float32[],
         input_id = Int[],
         temperature = 1.0f0,
         rng = rng_)
    end
end

function update_state(p::DecisionNode, ps, st)
    @unpack temperature, rng = st
    @unpack weight = ps

    priors = p.simplex(weight, temperature)

    input_id = map(axes(priors, 1)) do i
        dist = Categorical(priors[i, :])
        id = rand(rng, dist)
        ll = logpdf(dist, id)
        return id, ll
    end

    merge(st,
          (;
           input_id = reduce(vcat, map(first, input_id)),
           loglikelihood = reduce(vcat, map(last, input_id))))
end

function (l::DecisionNode)(x::AbstractArray, ps, st::NamedTuple)
    y = _apply_node(l, x, ps, st)
    return y, st
end

function (l::DecisionNode{true})(x::AbstractArray, ps, st::NamedTuple)
    y = _apply_node(l, x, ps, st)
    return vcat(y, x), st
end

function _apply_node(l::DecisionNode, x::AbstractMatrix, ps, st)::AbstractMatrix
    reduce(hcat, map(eachcol(x)) do xi
        _apply_node(l, xi, ps, st)
           end)
end

function _apply_node(l::DecisionNode, x::AbstractVector, ps, st)::Number
    @unpack input_id = st
    l.f(x[input_id]...)
end

Distributions.logpdf(::DecisionNode, ps, st)::Number = begin
    @unpack loglikelihood = st
    sum(loglikelihood)
end

Distributions.pdf(::DecisionNode, ps, st)::Number = begin
    @unpack loglikelihood = st
    prod(exp, loglikelihood, init = one(eltype(loglikelihood)))
end

function set_temperature(::DecisionNode, temperature, ps, st)
    merge(st, (; temperature = temperature))
end

get_temperature(::DecisionNode, ps, st) = st.temperature
