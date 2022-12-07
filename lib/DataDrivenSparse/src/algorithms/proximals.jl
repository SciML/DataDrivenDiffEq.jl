@inline function (s::AbstractProximalOperator)(x::AbstractArray, y::BitArray,
                                               λ::T) where {T <: Real}
    @assert size(y) == size(x)
    active_set!(y, s, x, λ)
    for i in eachindex(x)
        x[i] = y[i] ? x[i] : zero(T)
    end
    return
end

"""
$(TYPEDEF)
Proximal operator which implements the soft thresholding operator.

```julia
sign(x) * max(abs(x) - λ, 0)
```
See [by Zheng et. al., 2018](https://ieeexplore.ieee.org/document/8573778).
"""
struct SoftThreshold <: AbstractProximalOperator end;

@inline function active_set!(idx::BitArray, ::SoftThreshold, x::AbstractArray{T},
                             λ::T) where {T}
    @assert size(idx) == size(x)
    @inbounds foreach(eachindex(x)) do i
        idx[i] = abs(x[i]) > λ
    end
    return
end

@inline function (s::SoftThreshold)(x::AbstractArray, λ::T) where {T <: Real}
    for i in eachindex(x)
        x[i] = sign(x[i]) * max(abs(x[i]) - λ, zero(eltype(x)))
    end
    return
end

@inline function (s::SoftThreshold)(y::AbstractArray, x::AbstractArray,
                                    λ::T) where {T <: Real} 
    @assert size(y) == size(x)
    for i in eachindex(x)
        y[i] = sign(x[i]) * max(abs(x[i]) - λ, zero(eltype(x)))
    end
    return
end

"""
$(TYPEDEF)
Proximal operator which implements the hard thresholding operator.

```julia
abs(x) > sqrt(2*λ) ? x : 0
```
See [by Zheng et. al., 2018](https://ieeexplore.ieee.org/document/8573778).
"""
struct HardThreshold <: AbstractProximalOperator end;

@inline function active_set!(idx::BitArray, ::HardThreshold, x::AbstractArray,
                             λ::T) where {T}
    @assert size(idx) == size(x)
    @inbounds foreach(eachindex(x)) do i
        idx[i] = abs(x[i]) > sqrt(2 * λ)
    end
    return
end

@inline function (s::HardThreshold)(x::AbstractArray, λ::T) where {T <: Real}
    for i in eachindex(x)
        x[i] = abs(x[i]) > sqrt(2 * λ) ? x[i] : zero(eltype(x))
    end
    return
end

@inline function (s::HardThreshold)(y::AbstractArray, x::AbstractArray,
                                    λ::T) where {T <: Real}
    @assert all(size(y) .== size(x))
    for i in eachindex(x)
        y[i] = abs(x[i]) > sqrt(2 * λ) ? x[i] : zero(eltype(x))
    end
    return
end

"""
$(TYPEDEF)
Proximal operator which implements the (smoothly) clipped absolute deviation operator.

```julia
abs(x) > ρ ? x : sign(x) * max(abs(x) - λ, 0)
```

Where `ρ = 5λ` per default.

#Fields
$(FIELDS)

# Example

```julia
opt = ClippedAbsoluteDeviation()
opt = ClippedAbsoluteDeviation(1e-1)
```

See [by Zheng et. al., 2018](https://ieeexplore.ieee.org/document/8573778).
"""
struct ClippedAbsoluteDeviation{T} <: AbstractProximalOperator where {T <: Real}
    """Upper threshold"""
    ρ::T
end

ClippedAbsoluteDeviation() = ClippedAbsoluteDeviation(NaN)

@inline function active_set!(idx::BitArray, h::ClippedAbsoluteDeviation,
                             x::AbstractArray, λ::T) where {T}
    @assert size(idx) == size(x)
    @unpack ρ = h
    ρ = isnan(ρ) ? convert(T, 5) * λ : convert(T, ρ)
    @inbounds foreach(eachindex(x)) do i
        idx[i] = abs(x[i]) > ρ
    end
    return
end

function (s::ClippedAbsoluteDeviation)(x::AbstractArray, λ::T) where {T <: Real}
    @unpack ρ = h
    ρ = isnan(ρ) ? convert(T, 5) * λ : convert(T, ρ)
    for i in eachindex(x)
        x[i] = abs(x[i]) > ρ ? x[i] : sign(x[i]) * max(abs(x[i]) - λ, 0)
    end
    return
end

function (s::ClippedAbsoluteDeviation)(y::AbstractArray, x::AbstractArray,
                                       λ::T) where {T <: Real}
    @assert all(size(y) .== size(x))
    ρ = isnan(s.ρ) ? convert(eltype(x), 5) * λ : convert(eltype(x), s.ρ)
    for i in eachindex(x)
        y[i] = abs(x[i]) > ρ ? x[i] : sign(x[i]) * max(abs(x[i]) - λ, 0)
    end
    return
end
