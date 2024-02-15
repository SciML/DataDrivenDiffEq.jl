"""
$(TYPEDEF)
using Base: Multimedia

An error following `ŷ ~ y + ϵ`.
"""

struct AdditiveError <: AbstractErrorModel end

function (x::AdditiveError)(d::D, y::T, ỹ::R,
        scale::S = one(T)) where {D <: Type, T <: Number, S <: Number,
        R <: Number}
    logpdf(d(y, scale), ỹ)
end

"""
$(TYPEDEF)

An error following `ŷ ~ y * (1+ϵ)`.
"""

struct MultiplicativeError <: AbstractErrorModel end

function (x::MultiplicativeError)(d::D, y::T, ỹ::R,
        scale::S = one(T)) where {D <: Type, T <: Number,
        S <: Number, R <: Number}
    logpdf(d(y, abs(y) * scale), ỹ)
end

struct ObservedDistribution{fixed, D <: Distribution, M <: AbstractErrorModel, S, T}
    "The errormodel used for the output"
    errormodel::M
    "The (latent) scale parameter. If `fixed` this is equal to the true scale."
    latent_scale::S
    "The transformation used to transform the latent scale onto its domain"
    scale_transformation::T
end

function ObservedDistribution(distribution::Type{T}, errormodel::AbstractErrorModel;
        fixed = false,
        transform = as(Real, 1e-5, TransformVariables.∞),
        scale = 1.0) where {
        T <:
        Distributions.Distribution{Univariate,
        <:Any}}
    latent_scale = TransformVariables.inverse(transform, scale)
    return ObservedDistribution{fixed, T, typeof(errormodel), typeof(latent_scale),
        typeof(transform)}(errormodel, latent_scale, transform)
end

function Base.summary(io::IO, d::ObservedDistribution{fixed, D, E}) where {fixed, D, E}
    begin
        print(io, "$E : $D() with $(fixed ? "fixed" : "variable") scale.")
    end
end

get_init(d::ObservedDistribution) = d.latent_scale
function get_scale(d::ObservedDistribution)
    TransformVariables.transform(d.scale_transformation, d.latent_scale)
end
get_dist(d::ObservedDistribution{<:Any, D}) where {D} = D

Base.show(io::IO, d::ObservedDistribution) = summary(io, d)

function Distributions.logpdf(d::ObservedDistribution{false}, x::X, x̂::Y,
        scale::S) where {X, Y, S <: Number}
    sum(map(
        xs -> d.errormodel(
            get_dist(d), xs..., TransformVariables.transform(d.scale_transformation, scale)),
        zip(x, x̂)))
end

function Distributions.logpdf(d::ObservedDistribution{true}, x::X, x̂::Y,
        scale::S) where {X, Y, S <: Number}
    sum(map(
        xs -> d.errormodel(get_dist(d), xs...,
            TransformVariables.transform(d.scale_transformation, d.latent_scale)),
        zip(x, x̂)))
end

function Distributions.logpdf(d::ObservedDistribution{false}, x::X, x̂::Number,
        scale::S) where {X, S <: Number}
    sum(map(
        xs -> d.errormodel(get_dist(d), xs, x̂,
            TransformVariables.transform(d.scale_transformation, scale)),
        x))
end

function Distributions.logpdf(d::ObservedDistribution{true}, x::X, x̂::Number,
        scale::S) where {X, S <: Number}
    sum(map(
        xs -> d.errormodel(get_dist(d), xs, x̂,
            TransformVariables.transform(d.scale_transformation, d.latent_scale)),
        x))
end

function transform_scales(d::ObservedDistribution, scale::T) where {T <: Number}
    TransformVariables.transform(d.scale_transformation, scale)
end

"""
$(TYPEDEF)

The error distribution of a models output.
"""
struct ObservedModel{fixed, M}
    observed_distributions::NTuple{M, ObservedDistribution}
end

function ObservedModel(Y::AbstractMatrix; fixed = false)
    σ = ones(eltype(Y), size(Y, 1))
    dists = map(axes(Y, 1)) do i
        ObservedDistribution(Normal, AdditiveError(), fixed = fixed, scale = σ[i])
    end
    return ObservedModel{fixed, size(Y, 1)}(tuple(dists...))
end

needs_optimization(o::ObservedModel{fixed}) where {fixed} = !fixed

function Base.summary(io::IO, o::ObservedModel{<:Any, M}) where {M}
    print(io, "Observed Model with $M variables.")
end

Base.show(io::IO, o::ObservedModel) = summary(io, o)

function Distributions.logpdf(o::ObservedModel{M}, x::AbstractMatrix, x̂::AbstractMatrix,
        scales::AbstractVector = ones(eltype(x̂), size(x, 1))) where {
        M
}
    sum(map(logpdf, o.observed_distributions, eachrow(x), eachrow(x̂), scales))
end

function Distributions.logpdf(o::ObservedModel{M}, x::AbstractMatrix, x̂::AbstractVector,
        scales::AbstractVector = ones(eltype(x̂), size(x, 1))) where {
        M
}
    sum(map(axes(x, 1)) do i
        logpdf(o.observed_distributions[i], x[i, :], x̂[i], scales[i])
    end)
end

get_init(o::ObservedModel) = collect(map(get_init, o.observed_distributions))

function transform_scales(o::ObservedModel, latent_scales::AbstractVector)::AbstractVector
    collect(map(transform_scales, o.observed_distributions, latent_scales))
end

## Parameter Distributions
struct ParameterDistribution{P <: Distribution{Univariate}, T, I <: Interval, D <: Number}
    distribution::P
    interval::I
    transformation::T
    init::D
end

function ParameterDistribution(d::Distribution{Univariate}, init = mean(d),
        type::Type{T} = Float64) where {T}
    lower, upper = convert.(T, extrema(d))
    lower_t = isinf(lower) ? -TransformVariables.∞ : lower
    upper_t = isinf(upper) ? TransformVariables.∞ : upper
    transform = as(Real, lower_t, upper_t)
    init = convert.(T, TransformVariables.inverse(transform, init))
    return ParameterDistribution(d, Interval(lower, upper), transform, init)
end

function Base.summary(io::IO, p::ParameterDistribution)
    print(io, "$(p.distribution) distributed parameter ∈ $(p.interval)")
end
Base.show(io::IO, p::ParameterDistribution) = summary(io, p)

get_init(p::ParameterDistribution) = p.init
function transform_parameter(p::ParameterDistribution, pval::T) where {T <: Number}
    TransformVariables.transform(p.transformation, pval)
end
get_interval(p::ParameterDistribution) = p.interval

function Distributions.logpdf(p::ParameterDistribution, pval::T) where {T <: Number}
    transform_logdensity(p.transformation, Base.Fix1(logpdf, p.distribution), pval)
end

# Parameters 

struct ParameterDistributions{T, N}
    distributions::NTuple{N, ParameterDistribution}
end

function ParameterDistributions(b::Basis, eltype::Type{T} = Float64) where {T}
    isempty(ModelingToolkit.parameters(b)) &&
        return ParameterDistributions{T, 0}(NTuple{0, ParameterDistribution}())
    distributions = map(ModelingToolkit.parameters(b)) do p
        lower, upper = getbounds(p)
        dist = hasdist(p) ? getdist(p) : Uniform(lower, upper)

        # Check if we need to adjust the bounds
        if !Distributions.isbounded(dist)
            dist = truncated(dist, lower, upper)
        end

        if hasmetadata(p, Symbolics.VariableDefaultValue)
            init = Symbolics.getdefaultval(p)
        else
            init = Distributions.mean(dist)
        end
        ParameterDistribution(dist, init, T)
    end

    return ParameterDistributions{T, length(distributions)}(tuple(distributions...))
end

needs_optimization(::ParameterDistributions{<:Any, L}) where {L} = L > 0

function Base.summary(io::IO, p::ParameterDistributions)
    map(Base.Fix1(println, io), p.distributions)
end
Base.show(io::IO, p::ParameterDistributions) = summary(io, p)

get_init(p::ParameterDistributions) = collect(map(get_init, p.distributions))
function transform_parameter(p::ParameterDistributions, pval::P) where {P}
    collect(map(transform_parameter, p.distributions, pval))
end
get_interval(p::ParameterDistributions) = collect(map(get_interval, p.distributions))
function Distributions.logpdf(p::ParameterDistributions, pval::T) where {T}
    sum(map(logpdf, p.distributions, pval))
end

get_init(p::ParameterDistributions{T, 0}) where {T} = T[]
transform_parameter(p::ParameterDistributions{T, 0}, pval) where {T} = T[]
get_interval(p::ParameterDistributions{T, 0}) where {T} = Interval{T}[]
Distributions.logpdf(p::ParameterDistributions{T, 0}, pval) where {T} = zero(T)
