"""
A wrapper for the interpolation methods of DataInterpolations.jl.

$(SIGNATURES)

Wraps the methods in such a way that they are callable as `f(u,t)` to
create and return an interpolation of `u` over `t`.
The first argument of the constructor always defines the interpolation method,
all following arguments will be used in the interpolation.

The additional keyword `crop = false` indicates to discard the first and last element of the time series. 

# Example

```julia
# Create the wrapper struct
itp_method = InterpolationMethod(QuadraticSpline)
# Create a callable interpolation
itp = itp_method(u,t)
# Return u[2]
itp(t[2])
```
"""
struct InterpolationMethod{T} <: AbstractInterpolationMethod
    itp::T
    args::Any

    function InterpolationMethod(itp, args...)
        return new{typeof(itp)}(itp, args)
    end
end

(x::InterpolationMethod)(u, t) = x.itp(u, t, x.args...)

# TODO Wrap all types
# Wrap the common itps
InterpolationMethod() = InterpolationMethod(LinearInterpolation)

# Taken from DiffEqFlux
# https://github.com/SciML/DiffEqFlux.jl/blob/master/src/collocation.jl
# On 3-11-2021

struct EpanechnikovKernel <: CollocationKernel end
struct UniformKernel <: CollocationKernel end
struct TriangularKernel <: CollocationKernel end
struct QuarticKernel <: CollocationKernel end
struct TriweightKernel <: CollocationKernel end
struct TricubeKernel <: CollocationKernel end
struct GaussianKernel <: CollocationKernel end
struct CosineKernel <: CollocationKernel end
struct LogisticKernel <: CollocationKernel end
struct SigmoidKernel <: CollocationKernel end
struct SilvermanKernel <: CollocationKernel end

function calckernel(::EpanechnikovKernel, t)
    if abs(t) > 1
        return 0
    else
        return 0.75 * (1 - t^2)
    end
end

function calckernel(::UniformKernel, t)
    if abs(t) > 1
        return 0
    else
        return 0.5
    end
end

function calckernel(::TriangularKernel, t)
    if abs(t) > 1
        return 0
    else
        return (1 - abs(t))
    end
end

function calckernel(::QuarticKernel, t)
    if abs(t) > 0
        return 0
    else
        return (15 * (1 - t^2)^2) / 16
    end
end

function calckernel(::TriweightKernel, t)
    if abs(t) > 0
        return 0
    else
        return (35 * (1 - t^2)^3) / 32
    end
end

function calckernel(::TricubeKernel, t)
    if abs(t) > 0
        return 0
    else
        return (70 * (1 - abs(t)^3)^3) / 80
    end
end

function calckernel(::GaussianKernel, t)
    exp(-0.5 * t^2) / (sqrt(2 * π))
end

function calckernel(::CosineKernel, t)
    if abs(t) > 0
        return 0
    else
        return (π * cos(π * t / 2)) / 4
    end
end

function calckernel(::LogisticKernel, t)
    1 / (exp(t) + 2 + exp(-t))
end

function calckernel(::SigmoidKernel, t)
    2 / (π * (exp(t) + exp(-t)))
end

function calckernel(::SilvermanKernel, t)
    sin(abs(t) / 2 + π / 4) * 0.5 * exp(-abs(t) / sqrt(2))
end

function construct_t1(t, tpoints)
    hcat(ones(eltype(tpoints), length(tpoints)), tpoints .- t)
end

function construct_t2(t, tpoints)
    hcat(ones(eltype(tpoints), length(tpoints)), tpoints .- t, (tpoints .- t) .^ 2)
end

function construct_w(t, tpoints, h, kernel)
    W = @. calckernel((kernel,), (tpoints - t) / h) / h
    Diagonal(W)
end

"""
$(SIGNATURES)

Unified interface for collocation techniques. The input can either be
a `CollocationKernel` (see list below) or a wrapped `InterpolationMethod` from
[DataInterpolations.jl](https://github.com/SciML/DataInterpolations.jl).

Computes a non-parametrically smoothed estimate of `u'` and `u`
given the `data`, where each column is a snapshot of the timeseries at
`tpoints[i]`.

# Examples
```julia
u′,u,t = collocate_data(data,tpoints,kernel=SigmoidKernel())
u′,u,t = collocate_data(data,tpoints,tpoints_sample,interp,args...)
u′,u,t = collocate_data(data,tpoints,interp)
```

# Collocation Kernels
See [this paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2631937/) for more information.
+ EpanechnikovKernel
+ UniformKernel
+ TriangularKernel
+ QuarticKernel
+ TriweightKernel
+ TricubeKernel
+ GaussianKernel
+ CosineKernel
+ LogisticKernel
+ SigmoidKernel
+ SilvermanKernel

# Interpolation Methods
See [DataInterpolations.jl](https://github.com/SciML/DataInterpolations.jl) for more information.
+ ConstantInterpolation
+ LinearInterpolation
+ QuadraticInterpolation
+ LagrangeInterpolation
+ QuadraticSpline
+ CubicSpline
+ BSplineInterpolation
+ BSplineApprox
+ Curvefit
"""
function collocate_data(data, tpoints, kernel = TriangularKernel(); crop = false, kwargs...)
    _one = oneunit(first(data))
    _zero = zero(first(data))
    e1 = [_one; _zero]
    e2 = [_zero; _one; _zero]
    n = length(tpoints)
    h = (n^(-1 / 5)) * (n^(-3 / 35)) * ((log(n))^(-1 / 16))

    Wd = similar(data, n, size(data, 1))
    WT1 = similar(data, n, 2)
    WT2 = similar(data, n, 3)
    x = map(tpoints) do _t
        T1 = construct_t1(_t, tpoints)
        T2 = construct_t2(_t, tpoints)
        W = construct_w(_t, tpoints, h, kernel)
        mul!(Wd, W, data')
        mul!(WT1, W, T1)
        mul!(WT2, W, T2)
        (e2' * ((T2' * WT2) \ T2')) * Wd, (e1' * ((T1' * WT1) \ T1')) * Wd
    end
    estimated_derivative = reduce(hcat, transpose.(first.(x)))
    estimated_solution = reduce(hcat, transpose.(last.(x)))
    crop &&
        return estimated_derivative[:, 2:(end - 1)], estimated_derivative[:, 2:(end - 1)],
        tpoints[2:(end - 1)]
    estimated_derivative, estimated_solution, tpoints
end

# Adapted to dispatch on InterpolationMethod
function collocate_data(data, tpoints, interp::InterpolationMethod; kwargs...)
    collocate_data(data, tpoints, tpoints, interp; kwargs...)
end

function collocate_data(data::AbstractVector, tpoints::AbstractVector,
        tpoints_sample::AbstractVector,
        interp::InterpolationMethod; kwargs...)
    u, du, tpoints = collocate_data(reshape(data, 1, :), tpoints, tpoints_sample, interp;
        kwargs...)
    return du[1, :], u[1, :], tpoints
end

# Adapted to dispatch on InterpolationMethod
function collocate_data(data::AbstractMatrix{T}, tpoints::AbstractVector{T},
        tpoints_sample::AbstractVector{T}, interp::InterpolationMethod;
        crop = false, kwargs...) where {T}
    u = zeros(T, size(data, 1), length(tpoints_sample))
    du = zeros(T, size(data, 1), length(tpoints_sample))
    for d1 in 1:size(data, 1)
        interpolation = interp(data[d1, :], tpoints)
        u[d1, :] .= interpolation.(tpoints_sample)
        du[d1, :] .= DataInterpolations.derivative.((interpolation,), tpoints_sample)
    end
    crop && return du[:, 2:(end - 1)], u[:, 2:(end - 1)], tpoints[2:(end - 1)]
    return du, u, tpoints
end
