"""
$(TYPEDEF)

Maps an `AbstractArray` to the probability simplex by using `softmax`
on each row.
"""
struct Softmax <: AbstractSimplex end

function (::Softmax)(rng, x::AbstractArray, κ = one(eltype(x)))
    begin q = softmax(x ./ κ, dims = 2) end
end

"""
$(TYPEDEF)

Maps an `AbstractArray` to the probability simplex by adding gumbel distributed 
noise and using `softmax` on each row.

# Fields
$(FIELDS)
"""
struct GumbelSoftmax <: AbstractSimplex end

function (::GumbelSoftmax)(rng::Random.AbstractRNG, x::AbstractArray, κ = one(eltype(x)))
    z = -log.(-log.(rand(rng, size(x)...)))
    y = similar(x)
    foreach(axes(x, 2)) do i
        y[:, i] .= exp.(x[:, i])
    end
    y ./= sum(y, dims = 2)
    softmax((y .+ z) ./ κ, dims = 2)
end
