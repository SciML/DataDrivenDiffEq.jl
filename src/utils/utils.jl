# Optimal Shrinkage for data in presence of white noise
# See D. L. Donoho and M. Gavish, "The Optimal Hard Threshold for Singular
# Values is 4/sqrt(3)", http://arxiv.org/abs/1305.5870
# Code taken from https://github.com/erichson/optht

function optimal_svht(m::Int64, n::Int64; known_noise::Bool = false)
    @assert m / n > 0
    @assert m / n <= 1

    β = m / n
    ω = (8 * β) / (β + 1 + sqrt(β^2 + 14β + 1))
    c = sqrt(2 * (β + 1) + ω)

    if known_noise
        return c
    else
        median = median_marcenko_pastur(β)
        return c / sqrt(median)
    end
end

function marcenko_pastur_density(t, lower, upper, beta)
    sqrt((upper - t) .* (t - lower)) ./ (2π * beta * t)
end

function incremental_marcenko_pastur(x, beta, gamma)
    @assert beta <= 1
    upper = (1 + sqrt(beta))^2
    lower = (1 - sqrt(beta))^2

    @inline function marcenko_pastur(x)
        begin
            if (upper - x) * (x - lower) > 0
                return marcenko_pastur_density(x, lower, upper, beta)
            else
                return zero(eltype(x))
            end
        end
    end

    if gamma ≈ zero(eltype(gamma))
        i, ϵ = quadgk(x -> (x^gamma) * marcenko_pastur(x), x, upper)
        return i
    else
        i, ϵ = quadgk(x -> marcenko_pastur(x), x, upper)
        return i
    end
end

function median_marcenko_pastur(beta)
    @assert 0 < beta <= 1
    upper = (1 + sqrt(beta))^2
    lower = (1 - sqrt(beta))^2
    change = true
    x = ones(eltype(upper), 5)
    y = similar(x)
    while change && (upper - lower > 1e-5)
        x = range(lower, upper, length = 5)
        for (i, xi) in enumerate(x)
            y[i] = one(eltype(x)) - incremental_marcenko_pastur(xi, beta, 0)
        end
        any(y .< 0.5) ? lower = maximum(x[y .< 0.5]) : change = false
        any(y .> 0.5) ? upper = minimum(x[y .> 0.5]) : change = false
    end
    return (lower + upper) / 2
end

"""
    $(SIGNATURES)

Compute a feature reduced version of the data array `X` via thresholding the
singular values by computing the [optimal threshold for singular values](http://arxiv.org/abs/1305.5870).
"""
function optimal_shrinkage(X::AbstractArray{T, 2}) where {T <: Number}
    m, n = minimum(size(X)), maximum(size(X))
    U, S, V = svd(X)
    τ = optimal_svht(m, n)
    inds = S .>= τ * median(S)
    return U[:, inds] * Diagonal(S[inds]) * V[:, inds]'
end

"""
    $(SIGNATURES)

Compute a feature reduced version of the data array `X` inplace via thresholding the
singular values by computing the [optimal threshold for singular values](http://arxiv.org/abs/1305.5870).
"""
function optimal_shrinkage!(X::AbstractArray{T, 2}) where {T <: Number}
    m, n = minimum(size(X)), maximum(size(X))
    U, S, V = svd(X)
    τ = optimal_svht(m, n)
    inds = S .>= τ * median(S)
    X .= U[:, inds] * Diagonal(S[inds]) * V[:, inds]'
    return
end
