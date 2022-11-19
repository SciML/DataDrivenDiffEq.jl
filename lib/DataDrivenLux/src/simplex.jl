"""
$(TYPEDEF)

Maps an `AbstractArray` to the probability simplex by using `softmax`
on each row.
"""
struct Softmax <: AbstractSimplex end

(::Softmax)(x::AbstractArray, κ = one(eltype(x))) = softmax(x ./ κ, dims = 2)

"""
$(TYPEDEF)

Maps an `AbstractArray` to the probability simplex by adding gumbel distributed 
noise and using `softmax` on each row.

# Fields
$(FIELDS)
"""
struct GumbelSoftmax <: AbstractSimplex
    "Random number generator used for gumbel noise"
    rng::Random.AbstractRNG
end

function (::GumbelSoftmax)(x::AbstractArray, κ = one(eltype(x)))
    begin
        z = Zygote.@ignore -log.(-log.(rand(size(x)...)))
        y = similar(x)
        foreach(axes(x, 2)) do i
            y[:, i] .= exp.(x[:, i])
        end
        y ./= sum(y, dims = 2)
        softmax((y .+ z) ./ κ, dims = 2)
    end
end
