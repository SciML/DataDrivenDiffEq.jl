init_weights(::AbstractSimplex, rng::AbstractRNG, dims...) = zeros32(rng, dims...)

"""
$(TYPEDEF)

Maps an `AbstractVector` to the probability simplex by using `softmax`
on each row.
"""
struct Softmax <: AbstractSimplex end

function (::Softmax)(
        rng::AbstractRNG, x̂::AbstractVector, x::AbstractVector, κ = one(eltype(x)))
    return softmax!(x̂, x ./ κ)
end

"""
$(TYPEDEF)

Maps an `AbstractVector` to the probability simplex by adding gumbel distributed 
noise and using `softmax` on each row.

# Fields
$(FIELDS)
"""
struct GumbelSoftmax <: AbstractSimplex end

function (::GumbelSoftmax)(
        rng::AbstractRNG, x̂::AbstractVector, x::AbstractVector, κ = one(eltype(x)))
    z = -log.(-log.(rand(rng, size(x)...)))
    y = similar(x)
    foreach(axes(x, 2)) do i
        return y[:, i] .= exp.(x[:, i])
    end
    y ./= sum(y, dims = 2)
    return softmax!(x̂, (y .+ z) ./ κ)
end

"""
$(TYPEDEF)

Assumes an `AbstractVector` is on the probability simplex.

# Fields
$(FIELDS)
"""
struct DirectSimplex <: AbstractSimplex end

function (::DirectSimplex)(
        rng::AbstractRNG, x̂::AbstractVector, x::AbstractVector, κ = one(eltype(x)))
    return x̂ .= x
end

function init_weights(::DirectSimplex, rng::AbstractRNG, dims...)
    w = ones32(rng, dims...)
    w ./= first(dims)
    return w
end
