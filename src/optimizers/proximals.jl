"""
$(TYPEDEF)

Proximal operator which implements the soft thresholding operator.

See [by Zheng et. al., 2018](https://ieeexplore.ieee.org/document/8573778).
"""
struct SoftThreshold <: AbstractProximalOperator end;

@inline function (s::SoftThreshold)(x::AbstractArray, λ::T) where T <: Real
    for i in eachindex(x)
        x[i] = sign(x[i]) * max(abs(x[i]) - λ, zero(eltype(x)))
    end
    return
end

@inline function (s::SoftThreshold)(y::AbstractArray, x::AbstractArray, λ::T) where T <: Real
    @assert all(size(y) .== size(x))
    for i in eachindex(x)
        y[i] = sign(x[i]) * max(abs(x[i]) - λ, zero(eltype(x)))
    end
    return
end

"""
$(TYPEDEF)

Proximal operator which implements the hard thresholding operator.

See [by Zheng et. al., 2018](https://ieeexplore.ieee.org/document/8573778).
"""
struct HardThreshold <: AbstractProximalOperator end;

@inline function (s::HardThreshold)(x::AbstractArray, λ::T) where T <: Real
    for i in eachindex(x)
        x[i] = abs(x[i]) > sqrt(2*λ) ? x[i] : zero(eltype(x))
    end
    return
end

@inline function (s::HardThreshold)(y::AbstractArray, x::AbstractArray, λ::T) where T <: Real
    @assert all(size(y) .== size(x))
    for i in eachindex(x)
        y[i] = abs(x[i]) > sqrt(2*λ) ? x[i] : zero(eltype(x))
    end
    return
end
