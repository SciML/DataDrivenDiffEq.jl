function simplified_matvec(Ξ::AbstractArray{T, 2}, basis) where T <: Real
    eqs = Operation[]
    for i=1:size(Ξ, 2)
        eq = nothing
        for j = 1:size(Ξ, 1)
            if !iszero(Ξ[j,i])
                if eq === nothing
                    eq = basis[j]*Ξ[j,i]
                else
                    eq += basis[j]*Ξ[j,i]
                end
            end
        end
        if eq != nothing
            push!(eqs, eq)
        end
    end
    eqs
end

function simplified_matvec(Ξ::AbstractArray{T,1}, basis) where T <: Real
    eq = nothing
    @inbounds for i in 1:size(Ξ, 1)
        if !iszero(Ξ[i])
            if eq === nothing
                eq = basis[i]*Ξ[i]
            else
                eq += basis[i]*Ξ[i]
            end
        end

    end
    eq
end


function normalize_theta!(scales::AbstractArray, θ::AbstractArray)
    @assert length(scales) == size(θ, 1)
    @inbounds for (i, ti) in enumerate(eachrow(θ))
        scales[i] = norm(ti, 2)
        normalize!(ti, 2)
    end
    return
end

function rescale_xi!(scales::AbstractArray, Ξ::AbstractArray)
    @inbounds for (si, ti) in zip(scales, eachrow(Ξ))
        ti .= ti / si
    end
    return
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
function SInDy(X::AbstractArray{S, 2}, Ẋ::AbstractArray{S, 2}, Ψ::Basis; p::AbstractArray = [], maxiter::Int64 = 10, opt::T = Optimise.STRRidge(), denoise::Bool = false, normalize::Bool = true, return_coefficients::Bool = false) where {T <: Optimise.AbstractOptimiser, S <: Number}
    @assert size(X)[end] == size(Ẋ)[end]
    nx, nm = size(X)
    ny, nm = size(Ẋ)

    Ξ = zeros(eltype(X), length(Ψ), ny)
    scales = ones(eltype(X), length(Ψ))
    θ = Ψ(X, p = p)

    denoise ? optimal_shrinkage!(θ') : nothing
    normalize ? normalize_theta!(scales, θ) : nothing
    # Initial estimate
    Optimise.init!(Ξ, opt, θ', Ẋ')
    Optimise.fit!(Ξ, θ', Ẋ', opt, maxiter = maxiter)

    normalize ? rescale_xi!(scales, Ξ) : nothing

    if return_coefficients
        return Ξ'
    else
        return Basis(simplified_matvec(Ξ, Ψ.basis), variables(Ψ), parameters = p)
    end
end


function SInDy(X::AbstractArray{S, 1}, Ẋ::AbstractArray, Ψ::Basis, thresholds::AbstractArray; kwargs...) where S <: Number
    return SInDy(X', Ẋ, Ψ, thresholds; kwargs...)
end

function SInDy(X::AbstractArray{S, 2}, Ẋ::AbstractArray{S, 1}, Ψ::Basis, thresholds::AbstractArray; kwargs...) where S <: Number
    return SInDy(X, Ẋ', Ψ, thresholds; kwargs...)
end

function SInDy(X::AbstractArray{S, 2}, Ẋ::AbstractArray{S, 2}, Ψ::Basis, thresholds::AbstractArray ; p::AbstractArray = [], maxiter::Int64 = 10, opt::T = Optimise.STRRidge(),denoise::Bool = false, normalize::Bool = true, return_coefficients::Bool = false) where {T <: Optimise.AbstractOptimiser, S <: Number}
    @assert size(X)[end] == size(Ẋ)[end]
    nx, nm = size(X)
    ny, nm = size(Ẋ)

    θ = Ψ(X, p = p)

    ξ = zeros(eltype(X), length(Ψ), ny)
    Ξ_opt = zeros(eltype(X), length(Ψ), ny)
    Ξ = zeros(eltype(X), length(thresholds), ny, length(Ψ))
    x = zeros(eltype(X), length(thresholds), ny, 2)
    pareto = zeros(eltype(X),  ny, length(thresholds))
    scales = ones(eltype(X), length(Ψ))

    denoise ? optimal_shrinkage!(θ') : nothing
    normalize ? normalize_theta!(scales, θ) : nothing

    @inbounds for (j, threshold) in enumerate(thresholds)
        set_threshold!(opt, threshold)

        Optimise.init!(ξ, opt, θ', Ẋ')
        Optimise.fit!(ξ, θ', Ẋ', opt, maxiter = maxiter)

        [x[j, i, :] = [norm(xi, 0)/length(Ψ); norm(view(Ẋ , i, :) - θ'*xi, 2)] for (i, xi) in enumerate(eachcol(ξ))]

        normalize ? rescale_xi!(scales, ξ) : nothing
        Ξ[j, :, :] = ξ[:, :]'
    end

    # Create the evaluation
    @inbounds for i in 1:ny
        x[:, i, 2] .= x[:, i, 2]./maximum(x[:, i, 2])
        pareto[i, :] = [norm(x[j, i, :], 2) for j in 1:length(thresholds)]
        _, indx = findmin(pareto[i, :])
        Ξ_opt[:, i] = Ξ[indx, i, :]
    end

    if return_coefficients
        return Ξ'
    else
        return Basis(simplified_matvec(Ξ, Ψ.basis), variables(Ψ), parameters = p)
    end
end
