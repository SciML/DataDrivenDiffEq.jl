function simplified_matvec(Ξ, basis)
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

# Returns a basis for the differential state
function SInDy(X::AbstractArray, Ẋ::AbstractArray, Ψ::Basis; p::AbstractArray = [], maxiter::Int64 = 10, opt::T = Optimise.STRRidge()) where T <: Optimise.AbstractOptimiser
    θ = Ψ(X, p = p)
    # Initial estimate
    Ξ = Optimise.init(opt, θ', Ẋ')
    Optimise.fit!(Ξ, θ', Ẋ', opt, maxiter = maxiter)
    return Basis(simplified_matvec(Ξ, Ψ.basis), variables(Ψ), parameters = p)
end
