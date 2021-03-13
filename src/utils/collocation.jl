"""
A wrapper for the interpolation methods of DataInterpolations.jl.

$(SIGNATURES)

Wraps the methods in such a way that they are callable as `f(u,t)` to
create and return an interpolation of `u` over `t`.
The first argument of the constructor always defines the interpolation method,
all following arguments will be used in the interpolation.


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
  args

  function InterpolationMethod(itp, args...)
    return new{typeof(itp)}(itp, args)
  end

end

(x::InterpolationMethod)(u, t) = x.itp(u,t,x.args...)

# TODO Wrap all types
# Wrap the common itps
InterpolationMethod() = InterpolationMethod(QuadraticSpline)



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

function calckernel(::EpanechnikovKernel,t)
    if abs(t) > 1
        return 0
    else
        return 0.75*(1-t^2)
    end
end

function calckernel(::UniformKernel,t)
    if abs(t) > 1
        return 0
    else
        return 0.5
    end
end

function calckernel(::TriangularKernel,t)
    if abs(t) > 1
        return 0
    else
        return (1-abs(t))
    end
end

function calckernel(::QuarticKernel,t)
  if abs(t)>0
    return 0
  else
    return (15*(1-t^2)^2)/16
  end
end

function calckernel(::TriweightKernel,t)
  if abs(t)>0
    return 0
  else
    return (35*(1-t^2)^3)/32
  end
end

function calckernel(::TricubeKernel,t)
  if abs(t)>0
    return 0
  else
    return (70*(1-abs(t)^3)^3)/80
  end
end

function calckernel(::GaussianKernel,t)
  exp(-0.5*t^2)/(sqrt(2*π))
end

function calckernel(::CosineKernel,t)
  if abs(t)>0
    return 0
  else
    return (π*cos(π*t/2))/4
  end
end

function calckernel(::LogisticKernel,t)
  1/(exp(t)+2+exp(-t))
end

function calckernel(::SigmoidKernel,t)
  2/(π*(exp(t)+exp(-t)))
end

function calckernel(::SilvermanKernel,t)
  sin(abs(t)/2+π/4)*0.5*exp(-abs(t)/sqrt(2))
end

function construct_t1(t,tpoints)
    hcat(ones(eltype(tpoints),length(tpoints)),tpoints.-t)
end

function construct_t2(t,tpoints)
  hcat(ones(eltype(tpoints),length(tpoints)),tpoints.-t,(tpoints.-t).^2)
end

function construct_w(t,tpoints,h,kernel)
    W = @. calckernel((kernel,),(tpoints-t)/h)/h
    Diagonal(W)
end


"""
```julia
u′,u = collocate_data(data,tpoints,kernel=SigmoidKernel())
u′,u = collocate_data(data,tpoints,tpoints_sample,interp,args...)
```
Computes a non-parametrically smoothed estimate of `u'` and `u`
given the `data`, where each column is a snapshot of the timeseries at
`tpoints[i]`.
For kernels, the following exist:
- EpanechnikovKernel
- UniformKernel
- TriangularKernel
- QuarticKernel
- TriweightKernel
- TricubeKernel
- GaussianKernel
- CosineKernel
- LogisticKernel
- SigmoidKernel
- SilvermanKernel
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2631937/
Additionally, we can use interpolation methods from
[DataInterpolations.jl](https://github.com/PumasAI/DataInterpolations.jl) to generate
data from intermediate timesteps. In this case, pass any of the methods like
`QuadraticInterpolation` as `interp`, and the timestamps to sample from as `tpoints_sample`.
"""
function collocate_data(data,tpoints,kernel=TriangularKernel())
  _one = oneunit(first(data))
  _zero = zero(first(data))
  e1 = [_one;_zero]
  e2 = [_zero;_one;_zero]
  n = length(tpoints)
  h = (n^(-1/5))*(n^(-3/35))*((log(n))^(-1/16))

  Wd = similar(data, n, size(data,1))
  WT1 = similar(data, n, 2)
  WT2 = similar(data, n, 3)
  x = map(tpoints) do _t
      T1 = construct_t1(_t,tpoints)
      T2 = construct_t2(_t,tpoints)
      W = construct_w(_t,tpoints,h,kernel)
      mul!(Wd,W,data')
      mul!(WT1,W,T1)
      mul!(WT2,W,T2)
      (e2'*((T2'*WT2)\T2'))*Wd,(e1'*((T1'*WT1)\T1'))*Wd
  end
  estimated_derivative = reduce(hcat,transpose.(first.(x)))
  estimated_solution = reduce(hcat,transpose.(last.(x)))
  estimated_derivative,estimated_solution
end


# Adapted to dispatch on InterpolationMethod
collocate_data(data, tpoints, interp::InterpolationMethod) = collocate_data(data, tpoints, tpoints, interp)

function collocate_data(data::AbstractVector,tpoints::AbstractVector,tpoints_sample::AbstractVector,
                        interp::InterpolationMethod)
  u, du = collocate_data(reshape(data, 1, :),tpoints,tpoints_sample,interp)
  return du[1, :], u[1, :]
end

# Adapted to dispatch on InterpolationMethod
function collocate_data(data::AbstractMatrix{T},tpoints::AbstractVector{T},
                        tpoints_sample::AbstractVector{T},interp::InterpolationMethod) where T

  u = zeros(T,size(data, 1),length(tpoints_sample))
  du = zeros(T,size(data, 1),length(tpoints_sample))
  for d1 in 1:size(data,1)
    interpolation = interp(data[d1,:],tpoints)
    u[d1,:] .= interpolation.(tpoints_sample)
    du[d1,:] .= DataInterpolations.derivative.((interpolation,), tpoints_sample)
  end
  return du, u
end

# TODO Rework savitzky_golay to use the (u,t,args...) signature
# And use it as an InterpolationMethod
# Savitzky golay

#"""
#	savitzky_golay(X, windowSize, polyOrder; deriv, dt, crop)
#
#Estimate the time derivative via the savitzky_golay filter. `X` is the data matrix containing the trajectories, which is interpolated
#via polynomials of order `polyOrder` over `windowSize` points repeatedly. `deriv` defines the order of the derivative, `dt`
#the time step size. `crop` indicates if the original data should be returned cropped along the derivative approximation.
#"""
#function savitzky_golay(x::AbstractVector{T}, windowSize::Integer, polyOrder::Integer; deriv::Integer=0, dt::Real=1.0, crop::Bool = true) where T <: Number
#	# Polynomial smoothing with the Savitzky Golay filters
#	# Adapted from: https://github.com/BBN-Q/Qlab.jl/blob/master/src/SavitskyGolay.jl
#	# More information: https://pdfs.semanticscholar.org/066b/7534921b308925f6616480b4d2d2557943d1.pdf
#	# Requires LinearAlgebra and DSP modules loaded.
#
#	# Some error checking
#	@assert isodd(windowSize) "Window size must be an odd integer."
#	@assert polyOrder < windowSize "Polynomial order must be less than window size."
#
#	# Calculate filter coefficients
#	filterCoeffs = calculate_filterCoeffs(windowSize, polyOrder, deriv, dt)
#
#	# Pad the signal with the endpoints and convolve with filter
#	halfWindow = Int(ceil((windowSize - 1)/2))
#	paddedX = [x[1]*ones(halfWindow); x; x[end]*ones(halfWindow)]
#	y = conv(filterCoeffs[end:-1:1], paddedX)
#
#	if !crop
#		# Return the valid midsection
#		return y[2*halfWindow+1:end-2*halfWindow]
#	else
#		# Return cropped data. Excluding borders, where the estimation is less accurate
#		return x[halfWindow+2:end-halfWindow-1], y[3*halfWindow+2:end-3*halfWindow-1]
#	end
#end
#
#function savitzky_golay(x::AbstractMatrix{T}, windowSize::Integer, polyOrder::Integer; deriv::Integer=0, dt::Real=1.0, crop::Bool = true) where T <: Number
#	# Polynomial smoothing with the Savitzky Golay filters
#	# Adapted from: https://github.com/BBN-Q/Qlab.jl/blob/master/src/SavitskyGolay.jl
#	# More information: https://pdfs.semanticscholar.org/066b/7534921b308925f6616480b4d2d2557943d1.pdf
#	# Requires LinearAlgebra and DSP modules loaded.
#
#	# Some error checking
#	@assert isodd(windowSize) "Window size must be an odd integer."
#	@assert polyOrder < windowSize "Polynomial order must be less than window size."
#
#	# Calculate filter coefficients
#	filterCoeffs = calculate_filterCoeffs(windowSize, polyOrder, deriv, dt)
#
#	# Apply filter to each component
#	halfWindow = Int(ceil((windowSize - 1)/2))
#
#	if !crop
#		y = similar(x)
#		for (i, xi) in enumerate(eachrow(x))
#			paddedX = [xi[1]*ones(halfWindow); xi; xi[end]*ones(halfWindow)]
#			y₀ = conv(filterCoeffs[end:-1:1], paddedX)
#			y[i,:] = y₀[2*halfWindow+1:end-2*halfWindow]
#		end
#		return y
#	else
#		cropped_x = x[:,halfWindow+2:end-halfWindow-1]
#		y = similar(cropped_x)
#		for (i, xi) in enumerate(eachrow(x))
#			paddedX = [xi[1]*ones(halfWindow); xi; xi[end]*ones(halfWindow)]
#			y₀ = conv(filterCoeffs[end:-1:1], paddedX)
#			y[i,:] = y₀[3*halfWindow+2:end-3*halfWindow-1]
#		end
#		return cropped_x, y
#	end
#end
#
#function calculate_filterCoeffs(windowSize::Integer, polyOrder::Integer, deriv::Integer, dt::Real)
#	# Some error checking
#	@assert isodd(windowSize) "Window size must be an odd integer."
#	@assert polyOrder < windowSize "Polynomial order must be less than window size."
#
#	# Form the design matrix A
#	halfWindow = Int(ceil((windowSize - 1)/2))
#	A = zeros(windowSize, polyOrder+1)
#	for order = 0:polyOrder
#		A[:, order+1] = (-halfWindow:halfWindow).^(order)
#	end
#
#	# Compute the required column of the inverse of A'*A
#	# and calculate filter coefficients
#	ei = zeros(polyOrder+1)
#	ei[deriv+1] = 1.0
#	inv_col = (A'*A) \ ei
#	return A*inv_col * factorial(deriv) ./(dt^deriv)
#end
#
