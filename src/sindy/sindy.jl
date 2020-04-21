function normalize_theta!(scales::AbstractArray, θ::AbstractArray)
    @assert length(scales) == size(θ, 1)
    @inbounds for (i, ti) in enumerate(eachrow(θ))
        scales[i] = norm(ti, 2)
        normalize!(ti, 2)
    end
    return
end

function rescale_xi!(Ξ::AbstractArray, scales::AbstractArray)
    @assert length(scales) == size(Ξ, 1)
    @inbounds for (si, ti) in zip(scales, eachrow(Ξ))
        ti .= ti / si
    end
    return
end

function rescale_theta!(θ::AbstractArray, scales::AbstractArray)
    @assert length(scales) == size(θ, 1)
    @inbounds for (i, ti) in enumerate(eachrow(θ))
        ti .= ti * scales[i]
    end
    return
end

"""
    sparse_regression(X::AbstractArray, Y::AbstractArray, Ψ::Basis, p::AbstractArray, t::AbstractVector, maxiter::Int64, opt::AbstractOptimiser, denoise::Bool, normalize::Bool, convergence_error::Real)

    Sparsifying regression over the candidate library represented by the basis such that

"""
function sparse_regression(X::AbstractArray, Ẋ::AbstractArray, Ψ::Basis, p::AbstractArray, t::AbstractVector , maxiter::Int64 , opt::T, denoise::Bool, normalize::Bool, convergence_error) where T <: Optimise.AbstractOptimiser
    @assert size(X)[end] == size(Ẋ)[end]
    nx, nm = size(X)
    ny, nm = size(Ẋ)

    Ξ = zeros(eltype(X), length(Ψ), ny)
    scales = ones(eltype(X), length(Ψ))
    θ = Ψ(X, p, t)

    denoise ? optimal_shrinkage!(θ') : nothing
    normalize ? normalize_theta!(scales, θ) : nothing

    Optimise.init!(Ξ, opt, θ', Ẋ')
    iters = Optimise.fit!(Ξ, θ', Ẋ', opt, maxiter = maxiter, convergence_error = convergence_error)

    normalize ? rescale_xi!(Ξ, scales) : nothing

    return Ξ, iters
end

function sparse_regression!(Ξ::AbstractArray, X::AbstractArray, Ẋ::AbstractArray, Ψ::Basis, p::AbstractArray , t::AbstractVector, maxiter::Int64 , opt::T, denoise::Bool, normalize::Bool, convergence_error) where T <: Optimise.AbstractOptimiser
    @assert size(X)[end] == size(Ẋ)[end]
    nx, nm = size(X)
    ny, nm = size(Ẋ)
    @assert size(Ξ) == (length(Ψ), ny)

    scales = ones(eltype(X), length(Ψ))
    θ = Ψ(X, p, t)

    denoise ? optimal_shrinkage!(θ') : nothing
    normalize ? normalize_theta!(scales, θ) : nothing

    Optimise.init!(Ξ, opt, θ', Ẋ')
    iters = Optimise.fit!(Ξ, θ', Ẋ', opt, maxiter = maxiter, convergence_error = convergence_error)

    normalize ? rescale_xi!(Ξ, scales) : nothing

    return iters
end

# For pareto
function sparse_regression!(Ξ::AbstractArray, θ::AbstractArray, Ẋ::AbstractArray, maxiter::Int64 , opt::T, denoise::Bool, normalize::Bool, convergence_error) where T <: Optimise.AbstractOptimiser

    scales = ones(eltype(Ξ), size(θ, 1))

    denoise ? optimal_shrinkage!(θ') : nothing
    normalize ? normalize_theta!(scales, θ) : nothing

    Optimise.init!(Ξ, opt, θ', Ẋ')'
    iters = Optimise.fit!(Ξ, θ', Ẋ', opt, maxiter = maxiter, convergence_error = convergence_error)

    normalize ? rescale_xi!(Ξ, scales) : nothing
    normalize ? rescale_theta!(θ, scales) : nothing

    return iters
end


# One Variable on multiple derivatives
function SInDy(X::AbstractArray{S, 1}, Ẋ::AbstractArray, Ψ::Basis; kwargs...) where S <: Number
    return SInDy(X', Ẋ, Ψ; kwargs...)
end

# Multiple on one
function SInDy(X::AbstractArray{S, 2}, Ẋ::AbstractArray{S, 1}, Ψ::Basis; kwargs...) where S <: Number
    return SInDy(X, Ẋ', Ψ; kwargs...)
end

# General
function SInDy(X::AbstractArray{S, 2}, Ẋ::AbstractArray{S, 2}, Ψ::Basis; p::AbstractArray = [], t::AbstractVector = [], maxiter::Int64 = 10, opt::T = Optimise.STRRidge(), denoise::Bool = false, normalize::Bool = true, convergence_error = eps()) where {T <: Optimise.AbstractOptimiser, S <: Number}
    Ξ, iters = sparse_regression(X, Ẋ, Ψ, p, t, maxiter, opt, denoise, normalize, convergence_error)
    convergence = iters < maxiter
    SparseIdentificationResult(Ξ, Ψ, iters, opt, convergence, Ẋ, X, p = p)
end



function SInDy(X::AbstractArray{S, 1}, Ẋ::AbstractArray, Ψ::Basis, thresholds::AbstractArray; kwargs...) where S <: Number
    return SInDy(X', Ẋ, Ψ, thresholds; kwargs...)
end

function SInDy(X::AbstractArray{S, 2}, Ẋ::AbstractArray{S, 1}, Ψ::Basis, thresholds::AbstractArray; kwargs...) where S <: Number
    return SInDy(X, Ẋ', Ψ, thresholds; kwargs...)
end

function SInDy(X::AbstractArray{S, 2}, Ẋ::AbstractArray{S, 2}, Ψ::Basis, thresholds::AbstractArray ; weights::AbstractArray = [], f_target = (x, w) ->  norm(w .* x, 2),  p::AbstractArray = [], t::AbstractVector = [], maxiter::Int64 = 10, opt::T = Optimise.STRRidge(),denoise::Bool = false, normalize::Bool = true, convergence_error = eps()) where {T <: Optimise.AbstractOptimiser, S <: Number}
    @assert size(X)[end] == size(Ẋ)[end]
    nx, nm = size(X)
    ny, nm = size(Ẋ)

    θ = Ψ(X, p, t)

    scales = ones(eltype(X), length(Ψ))
    ξ = zeros(eltype(X), length(Ψ), ny)
    Ξ_opt = zeros(eltype(X), length(Ψ), ny)
    Ξ = zeros(eltype(X), length(thresholds), ny, length(Ψ))
    x = zeros(eltype(X), length(thresholds), ny, 2)
    iters = zeros(Int64, length(thresholds))

    denoise ? optimal_shrinkage!(θ') : nothing
    normalize ? normalize_theta!(scales, θ) : nothing

    @inbounds for (j, threshold) in enumerate(thresholds)
        set_threshold!(opt, threshold)

        iters[j] = sparse_regression!(ξ, θ, Ẋ, maxiter, opt, false, false, convergence_error)

        normalize ? rescale_xi!(ξ, scales) : nothing

        [x[j, i, :] = [norm(xi, 0); norm(view(Ẋ , i, :) - θ'*xi, 2)] for (i, xi) in enumerate(eachcol(ξ))]

        Ξ[j, :, :] = ξ[:, :]'
    end

    # Closure
    isempty(weights) ? weights = ones(eltype(x), 2)/2 : nothing
    f_t(x) = f_target(x, weights)

    _iter = Inf
    _thresh = Inf
    # Create the evaluation
    @inbounds for i in 1:ny
        _, idx = findmin(f_t.(eachrow(x[:, i, :])))
        iters[idx] <  _iter ? _iter = iters[idx] : nothing
        thresholds[idx] < _thresh ? _thresh = thresholds[idx] : nothing
        Ξ_opt[:, i] = Ξ[idx, i, :]
    end

    set_threshold!(opt, _thresh)

    return SparseIdentificationResult(Ξ_opt, Ψ, _iter, opt, _iter < maxiter, Ẋ, X, p = p)
end
