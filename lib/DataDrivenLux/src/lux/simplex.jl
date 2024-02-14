function init_weights(::AbstractSimplex, rng::Random.AbstractRNG, dims...)
    Lux.zeros32(rng, dims...)
end

"""
$(TYPEDEF)

Maps an `AbstractVector` to the probability simplex by using `softmax`
on each row.
"""
struct Softmax <: AbstractSimplex end

function (::Softmax)(rng::Random.AbstractRNG, x̂::AbstractVector, x::AbstractVector,
        κ = one(eltype(x)))
    softmax!(x̂, x ./ κ)
end

"""
$(TYPEDEF)

Maps an `AbstractVector` to the probability simplex by adding gumbel distributed 
noise and using `softmax` on each row.

# Fields
$(FIELDS)
"""
struct GumbelSoftmax <: AbstractSimplex end

function (::GumbelSoftmax)(rng::Random.AbstractRNG, x̂::AbstractVector, x::AbstractVector,
        κ = one(eltype(x)))
    z = -log.(-log.(rand(rng, size(x)...)))
    y = similar(x)
    foreach(axes(x, 2)) do i
        y[:, i] .= exp.(x[:, i])
    end
    y ./= sum(y, dims = 2)
    softmax!(x̂, (y .+ z) ./ κ)
end

"""
$(TYPEDEF)

Assumes an `AbstractVector` is on the probability simplex.

# Fields
$(FIELDS)
"""
struct DirectSimplex <: AbstractSimplex end

function (::DirectSimplex)(rng::Random.AbstractRNG, x̂::AbstractVector, x::AbstractVector,
        κ = one(eltype(x)))
    x̂ .= x
end

function init_weights(::DirectSimplex, rng::Random.AbstractRNG, dims...)
    w = Lux.ones32(rng, dims...)
    w ./= first(dims)
    w
end
