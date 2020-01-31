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
    θ = Ψ(X, p = p)
    # Initial estimate
    Ξ = Optimise.init(opt, θ', Ẋ')
    Optimise.fit!(Ξ, θ', Ẋ', opt, maxiter = maxiter)
    return Basis(simplified_matvec(Ξ, Ψ.basis), variables(Ψ), parameters = p)
end

# Returns an array of basis for all values of lambda
function SInDy(X::AbstractArray, Ẋ::AbstractArray, Ψ::Basis, thresholds::AbstractArray ; p::AbstractArray = [], maxiter::Int64 = 10, opt::T = Optimise.STRRidge()) where T <: Optimise.AbstractOptimiser
    basis = Basis[]
    for threshold in thresholds
        try
            push!(basis, SInDy(X, Ẋ, Ψ, p = p, maxiter = maxiter, opt = opt))
        catch
            println("Failed for threshold $threshold")
        end
    end
    return basis
end
