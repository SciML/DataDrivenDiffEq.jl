"""
$(TYPEDEF)
using Base: Multimedia

An error following `ŷ ~ y + ϵ`.
"""
struct AdditiveError <: AbstractErrorModel end

function (x::AdditiveError)(d::D, y::T, ỹ::T, scale::S = one(T)) where {D, T <: Number, S}
    begin logpdf(d(y, scale), ỹ) end
end

"""
$(TYPEDEF)

An error following `ŷ ~ y * (1+ϵ)`.
"""
struct MultiplicativeError <: AbstractErrorModel end

function (x::MultiplicativeError)(d::D, y::T, ỹ::T,
                                  scale::S = one(T)) where {D, T <: Number, S}
    logpdf(D(y, abs(y) * scale), ỹ)
end

"""
$(TYPEDEF)

The error distribution of a models output.
"""
struct ObservedError{D, E} <: AbstractErrorModel
    distributions::D
    error_models::E
end

function ObservedError(n::Int)
    distributions = Tuple(Normal for _ in 1:n)
    errors = Tuple(AdditiveError() for _ in 1:n)
    return ObservedError(distributions, errors)
end

function Distributions.logpdf(o::ObservedError{U, V}, y::AbstractMatrix{T},
                              ỹ::AbstractMatrix{T},
                              scales::AbstractVector{S} = ones(eltype(y), size(y, 1))) where {
                                                                                              U,
                                                                                              V,
                                                                                              T,
                                                                                              S
                                                                                              }
    @unpack distributions, error_models = o

    lpdf = zero(S)
    @inbounds for i in axes(y, 1), j in axes(y, 2)
        lpdf += error_models[i](distributions[i], y[i, j], ỹ[i, j], scales[i])
    end
    lpdf
end

function Distributions.logpdf(o::ObservedError{U, V}, y::AbstractMatrix{T},
                              ỹ::AbstractMatrix{T},
                              scales::AbstractMatrix{S}) where {U, V, T, S}
    @unpack distributions, error_models = o

    lpdf = zero(S)
    @inbounds for i in axes(y, 1), j in axes(y, 2)
        lpdf += error_models[i](distributions[i], y[i, j], ỹ[i, j], scales[i, j])
    end
    lpdf
end
