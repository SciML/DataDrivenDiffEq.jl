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
    S[S .< τ*median(S)] .= 0
    return U*Diagonal(S)*V'
end

function optimal_shrinkage!(X::AbstractArray{T, 2}) where T <: Number
    m,n = minimum(size(X)), maximum(size(X))
    U, S, V = svd(X)
    τ = optimal_svht(m,n)
    S[S .< τ*median(S)] .= 0
    X .= U*Diagonal(S)*V'
    return
end

function hankel(a::AbstractArray, b::AbstractArray) where T <: Number
    p = vcat([a; b[2:end]])
    n = length(b)-1
    m = length(p)-n+1
    H = Array{eltype(p)}(undef, n, m)
    @inbounds for i in 1:n
        for j in 1:m
            H[i,j] = p[i+j-1]
        end
    end
    return H
end

@inline function burst_sampling(x::AbstractArray, samplesize::Int64, bursts::Int64)
    @assert size(x)[end] >= samplesize*bursts "Length of data array too small for subsampling of size $size!"
    inds = sample(1:size(x)[end]-samplesize, bursts, replace = false) # Starting points
    inds = sort(unique(vcat([collect(i:i+samplesize) for i in inds]...)))
    return resample(x, inds)
end


@inline function burst_sampling(x::AbstractArray, y::AbstractArray, samplesize::Int64, bursts::Int64)
    @assert size(x)[end] >= samplesize*bursts "Length of data array too small for subsampling of size $size!"
    @assert size(x)[end] == size(y)[end]
    inds = sample(1:size(x)[end]-samplesize, bursts, replace = false) # Starting points
    inds = sort(unique(vcat([collect(i:i+samplesize) for i in inds]...)))
    return resample(x, inds), resample(y, inds)
end


@inline function burst_sampling(x::AbstractArray, t::AbstractVector, period::T, bursts::Int64) where T <: AbstractFloat
    @assert size(x)[end] == size(t)[end]
    @assert bursts >= 1
    @assert t[end]-t[1]>= period*bursts "Bursting impossible. Please provide more data or reduce bursts."
    t_ids = zero(eltype(t)) .<= t .- period  .<= t[end] .- 2*period
    samplesize = Int64(round(period/(t[end]-t[1])*length(t)))
    inds = sample(t_ids, bursts, replace = false)
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
