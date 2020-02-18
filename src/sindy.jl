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

# Returns a basis for the differential state
function SInDy(X::AbstractArray, Ẋ::AbstractArray, Ψ::Basis; p::AbstractArray = [], maxiter::Int64 = 10, opt::T = Optimise.STRRidge()) where T <: Optimise.AbstractOptimiser
    @assert size(X)[end] == size(Ẋ)[end]
    nx, nm = size(X)

    Ξ = zeros(eltype(X), length(Ψ), nx)
    θ = Ψ(X, p = p)

    # Initial estimate
    Optimise.init!(Ξ, opt, θ', Ẋ')
    Optimise.fit!(Ξ, θ', Ẋ', opt, maxiter = maxiter)
    return Basis(simplified_matvec(Ξ, Ψ.basis), variables(Ψ), parameters = p)
end


# Returns an array of basis for all values of lambda
function SInDy(X::AbstractArray, Ẋ::AbstractArray, Ψ::Basis, thresholds::AbstractArray ; p::AbstractArray = [], maxiter::Int64 = 10, opt::T = Optimise.STRRidge()) where T <: Optimise.AbstractOptimiser
    @assert size(X)[end] == size(Ẋ)[end]
    nx, nm = size(X)

    θ = Ψ(X, p = p)

    ξ = zeros(eltype(X), length(Ψ), nx)
    Ξ_opt = zeros(eltype(X), length(Ψ), nx)
    Ξ = zeros(eltype(X), length(thresholds), nx, length(Ψ))
    x = zeros(eltype(X), length(thresholds), nx, 2)
    p = zeros(eltype(X),  nx, length(thresholds))

    @inbounds for (j, threshold) in enumerate(thresholds)
        set_threshold!(opt, threshold)
        Optimise.init!(ξ, opt, θ', Ẋ')
        Optimise.fit!(ξ, θ', Ẋ', opt, maxiter = maxiter)
        Ξ[j, :, :] = ξ[:, :]'
        [x[j, i, :] = [norm(xi, 0)/length(Ψ); norm(view(Ẋ , i, :) - θ'*xi, 2)] for (i, xi) in enumerate(eachcol(ξ))]
    end

    # Create the evaluation
    @inbounds for i in 1:nx
        x[:, i, 2] .= x[:, i, 2]./maximum(x[:, i, 2])
        p[i, :] = [norm(x[j, i, :], 2) for j in 1:length(thresholds)]
        _, indx = findmin(p[i, :])
        Ξ_opt[:, i] = Ξ[indx, i, :]
    end

    return Basis(simplified_matvec(Ξ_opt, Ψ.basis), variables(Ψ), parameters = p)
end
