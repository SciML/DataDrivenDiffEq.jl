"""
$(TYPEDEF)
using Base: Multimedia

An error following `ŷ ~ y + ϵ`.
"""
struct AdditiveError <: AbstractErrorModel end

function (x::AdditiveError)(y::T, ỹ::T) where {T <: AbstractArray}
    @assert size(y) == size(ỹ) "Data and prediction have to be of equal size!"
    map(enumerate(y)) do (i, yi) 
        ỹ[i] - yi
    end
end

(x::AdditiveError)(y::T, ỹ::T) where {T <: Number} = ỹ - y

"""
$(TYPEDEF)

An error following `ŷ ~ y * (1+ϵ)`.
"""
struct MultiplicativeError <: AbstractErrorModel end

function (x::MultiplicativeError)(y::T, ỹ::T) where {T <: AbstractArray}
    @assert size(y) == size(ỹ) "Data and prediction have to be of equal size!"
    xone = one(eltype(y))
    map(enumerate(y)) do (i,yi) 
        xone - ỹ[i] / (yi .+ eps())
    end
end

(x::MultiplicativeError)(y::T, ỹ::T) where {T <: Number} = ỹ / (y + eps())


"""
$(TYPEDEF)

The error distribution of a models output.
"""
struct ObservedError{D, E} <: AbstractErrorModel 
    distributions::D
    error_models::E
end

function ObservedError(n::Int)
    distributions = Tuple(Normal() for _ in 1:n)
    errors = Tuple(AdditiveError() for _ in 1:n)
    return ObservedError(distributions, errors)
end

function Distributions.logpdf(o::ObservedError{U, V}, y::AbstractMatrix, ỹ::AbstractMatrix, scales::AbstractVector = ones(eltype(y), size(y, 1))) where {U, V} 
    @unpack distributions, error_models = o

    lpdf = zero(eltype(y))
    for i in axes(y, 1), j in axes(y, 2)
        lpdf += Distributions.logpdf(typeof(distributions[i])(zero(scales[i]), scales[i]), error_models[i](y[i, j], ỹ[i,j]))
    end
    lpdf 
end

function Distributions.logpdf(o::ObservedError{U, V}, y::AbstractMatrix, ỹ::AbstractMatrix, scales::AbstractMatrix) where {U, V} 
    @unpack distributions, error_models = o

    lpdf = zero(eltype(y))
    for i in axes(y, 1), j in axes(y, 2)
        lpdf += Distributions.logpdf(typeof(distributions[i])(zero(scales[i,j]), scales[i,j]), error_models[i](y[i, j], ỹ[i,j]))
    end
    lpdf
end