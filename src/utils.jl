import StatsBase: sample

# Model selection

# Taken from https://royalsocietypublishing.org/doi/pdf/10.1098/rspa.2017.0009
function AIC(k::Int64, X::AbstractArray, Y::AbstractArray; likelyhood = (X,Y) -> sum(abs2, X-Y))
    @assert size(X) == size(Y) "Dimensions of trajectories should be equal !"
    return 2*k - 2*log(likelyhood(X, Y))
end
# Taken from https://royalsocietypublishing.org/doi/pdf/10.1098/rspa.2017.0009
function AICC(k::Int64, X::AbstractArray, Y::AbstractArray; likelyhood = (X,Y) -> sum(abs2, X-Y))
    @assert size(X) == size(Y) "Dimensions of trajectories should be equal !"
    return AIC(k, X, Y, likelyhood = likelyhood)+ 2*(k+1)*(k+2)/(size(X)[2]-k-2)
end

# Double check on that
# Taken from https://www.immagic.com/eLibrary/ARCHIVES/GENERAL/WIKIPEDI/W120607B.pdf
function BIC(k::Int64, X::AbstractArray, Y::AbstractArray; likelyhood = (X,Y) -> sum(abs2, X-Y))
    @assert size(X) == size(Y) "Dimensions of trajectories should be equal !"
    return - 2*log(likelyhood(X, Y)) + k*log(size(X)[2])
end


# Optimal Shrinkage for data in presence of white noise
# See D. L. Donoho and M. Gavish, "The Optimal Hard Threshold for Singular
# Values is 4/sqrt(3)", http://arxiv.org/abs/1305.5870
# Code taken from https://github.com/erichson/optht

function optimal_svht(m::Int64, n::Int64; known_noise::Bool = false)
    @assert m/n > 0
    @assert m/n <= 1

    β = m/n
    ω = (8*β) / (β+1+sqrt(β^2+14β+1))
    c = sqrt(2*(β+1)+ω)

    if known_noise
        return c
    else
        median = median_marcenko_pastur(β)
        return c / sqrt(median)
    end
end

function marcenko_pastur_density(t, lower, upper, beta)
    sqrt((upper-t).*(t-lower))./(2π*beta*t)
end

function incremental_marcenko_pastur(x, beta, gamma)
    @assert beta <= 1
    upper = (1+sqrt(beta))^2
    lower = (1-sqrt(beta))^2

    @inline marcenko_pastur(x) = begin
        if (upper-x)*(x-lower) > 0
            return marcenko_pastur_density(x, lower, upper, beta)
        else
            return zero(eltype(x))
        end
    end

    if gamma ≈ zero(eltype(gamma))
        i, ϵ = quadgk(x->(x^gamma)*marcenko_pastur(x), x, upper)
        return i
    else
        i, ϵ = quadgk(x->marcenko_pastur(x), x, upper)
        return i
    end
end

function median_marcenko_pastur(beta)
    @assert 0 < beta <= 1
    upper = (1+sqrt(beta))^2
    lower = (1-sqrt(beta))^2
    change = true
    x = ones(eltype(upper), 5)
    y = similar(x)
    while change && (upper - lower > 1e-5)
        x = range(lower, upper, length = 5)
        for (i,xi) in enumerate(x)
            y[i] = one(eltype(x)) - incremental_marcenko_pastur(xi, beta, 0)
        end
        any(y .< 0.5) ? lower = maximum(x[y .< 0.5]) : change = false
        any(y .> 0.5) ? upper = minimum(x[y .> 0.5]) : change = false
    end
    return (lower+upper)/2
end

function optimal_shrinkage(X::AbstractArray{T, 2}) where T <: Number
    m,n = minimum(size(X)), maximum(size(X))
    U, S, V = svd(X)
    τ = optimal_svht(m,n)
    inds = S .>= τ*median(S)
    return U[:, inds]*Diagonal(S[inds])*V[:, inds]'
end

function optimal_shrinkage!(X::AbstractArray{T, 2}) where T <: Number
    m,n = minimum(size(X)), maximum(size(X))
    U, S, V = svd(X)
    τ = optimal_svht(m,n)
    inds = S .>= τ*median(S)
    X .= U[:, inds]*Diagonal(S[inds])*V[:, inds]'
    return
end

function savitzky_golay(x::AbstractVector{T}, windowSize::Integer, polyOrder::Integer; deriv::Integer=0, dt::Real=1.0, crop::Bool = true) where T <: Number
	# Polynomial smoothing with the Savitzky Golay filters
	# Adapted from: https://github.com/BBN-Q/Qlab.jl/blob/master/src/SavitskyGolay.jl
	# More information: https://pdfs.semanticscholar.org/066b/7534921b308925f6616480b4d2d2557943d1.pdf
	# Requires LinearAlgebra and DSP modules loaded.

	# Some error checking
	@assert isodd(windowSize) "Window size must be an odd integer."
	@assert polyOrder < windowSize "Polynomial order must be less than window size."

	# Calculate filter coefficients
	filterCoeffs = calculate_filterCoeffs(windowSize, polyOrder, deriv, dt)

	# Pad the signal with the endpoints and convolve with filter
	halfWindow = Int(ceil((windowSize - 1)/2))
	paddedX = [x[1]*ones(halfWindow); x; x[end]*ones(halfWindow)]
	y = conv(filterCoeffs[end:-1:1], paddedX)

	if !crop
		# Return the valid midsection
		return y[2*halfWindow+1:end-2*halfWindow]
	else
		# Return cropped data. Excluding borders, where the estimation is less accurate
		return x[halfWindow+2:end-halfWindow-1], y[3*halfWindow+2:end-3*halfWindow-1]
	end
end

function savitzky_golay(x::AbstractMatrix{T}, windowSize::Integer, polyOrder::Integer; deriv::Integer=0, dt::Real=1.0, crop::Bool = true) where T <: Number
	# Polynomial smoothing with the Savitzky Golay filters
	# Adapted from: https://github.com/BBN-Q/Qlab.jl/blob/master/src/SavitskyGolay.jl
	# More information: https://pdfs.semanticscholar.org/066b/7534921b308925f6616480b4d2d2557943d1.pdf
	# Requires LinearAlgebra and DSP modules loaded.

	# Some error checking
	@assert isodd(windowSize) "Window size must be an odd integer."
	@assert polyOrder < windowSize "Polynomial order must be less than window size."

	# Calculate filter coefficients
	filterCoeffs = calculate_filterCoeffs(windowSize, polyOrder, deriv, dt)

	# Apply filter to each component
	halfWindow = Int(ceil((windowSize - 1)/2))

	if !crop
		y = similar(x)
		for (i, xi) in enumerate(eachrow(x))
			paddedX = [xi[1]*ones(halfWindow); xi; xi[end]*ones(halfWindow)]
			y₀ = conv(filterCoeffs[end:-1:1], paddedX)
			y[i,:] = y₀[2*halfWindow+1:end-2*halfWindow]
		end
		return y
	else
		cropped_x = x[:,halfWindow+2:end-halfWindow-1]
		y = similar(cropped_x)
		for (i, xi) in enumerate(eachrow(x))
			paddedX = [xi[1]*ones(halfWindow); xi; xi[end]*ones(halfWindow)]
			y₀ = conv(filterCoeffs[end:-1:1], paddedX)
			y[i,:] = y₀[3*halfWindow+2:end-3*halfWindow-1]
		end
		return cropped_x, y
	end
end

function calculate_filterCoeffs(windowSize::Integer, polyOrder::Integer, deriv::Integer, dt::Real)
	# Some error checking
	@assert isodd(windowSize) "Window size must be an odd integer."
	@assert polyOrder < windowSize "Polynomial order must be less than window size."

	# Form the design matrix A
	halfWindow = Int(ceil((windowSize - 1)/2))
	A = zeros(windowSize, polyOrder+1)
	for order = 0:polyOrder
		A[:, order+1] = (-halfWindow:halfWindow).^(order)
	end

	# Compute the required column of the inverse of A'*A
	# and calculate filter coefficients
	ei = zeros(polyOrder+1)
	ei[deriv+1] = 1.0
	inv_col = (A'*A) \ ei
	return A*inv_col * factorial(deriv) ./(dt^deriv)
end

@inline function burst_sampling(x::AbstractArray, samplesize::Int64, bursts::Int64)
    @assert size(x)[end] >= samplesize*bursts "Length of data array too small for subsampling of size $size!"
    inds = sample(1:size(x)[end]-samplesize, bursts, replace = false)
    inds = sort(unique(vcat([collect(i:i+samplesize) for i in inds]...)))
    return resample(x, inds)
end


@inline function burst_sampling(x::AbstractArray, y::AbstractArray, samplesize::Int64, bursts::Int64)
    @assert size(x)[end] >= samplesize*bursts "Length of data array too small for subsampling of size $size!"
    @assert size(x)[end] == size(y)[end]
    inds = sample(1:size(x)[end]-samplesize, bursts, replace = false)
    inds = sort(unique(vcat([collect(i:i+samplesize) for i in inds]...)))
    return resample(x, inds), resample(y, inds)
end


@inline function burst_sampling(x::AbstractArray, t::AbstractVector, period::T, bursts::Int64) where T <: AbstractFloat
    @assert period > zero(typeof(period)) "Sampling period has to be positive."
    @assert size(x)[end] == size(t)[end] "Provide consistent data."
    @assert bursts >= 1 "Number of bursts has to be positive."
    @assert t[end]-t[1]>= period*bursts "Bursting impossible. Please provide more data or reduce bursts."
    t_ids = zero(eltype(t)) .<= t .- period  .<= t[end] .- 2*period
    samplesize = Int64(floor(period/(t[end]-t[1])*length(t)))
    inds = sample(collect(1:length(t))[t_ids], bursts, replace = false)
    inds = sort(unique(vcat([collect(i:i+samplesize) for i in inds]...)))
    return resample(x, inds), resample(t, inds)
end


@inline function subsample(x::AbstractVector, frequency::Int64)
    @assert frequency > 1
    return x[1:frequency:end]
end


@inline function subsample(x::AbstractArray, frequency::Int64)
    @assert frequency > 1
    return x[:, 1:frequency:end]
end

@inline function subsample(x::AbstractArray, t::AbstractVector, period::T) where T <: AbstractFloat
    @assert period > zero(typeof(period)) "Sampling period has to be positive."
    @assert size(x)[end] == size(t)[end] "Provide consistent data."
    @assert t[end]-t[1]>= period "Subsampling impossible. Sampling period exceeds time window."
    idx = Int64[1]
    t_now = t[1]
    @inbounds for (i, t_current) in enumerate(t)
        if t_current - t_now >= period
            push!(idx, i)
            t_now = t_current
        end
    end
    return resample(x, idx), resample(t, idx)
end

@inline function resample(x::AbstractArray{T,1}, indx::AbstractArray{Int64}) where T <: Number
    @assert maximum(indx) <= length(x)
    @assert minimum(indx) >= 1
    return x[indx]
end

@inline function resample(x::AbstractArray{T,2}, indx::AbstractArray{Int64}) where T <: Number
    @assert maximum(indx) <= size(x, 2)
    @assert minimum(indx) >= 1
    return x[:, indx]
end
