function STRridge(A::AbstractArray, Y::AbstractArray; ϵ::Number = 1e-3, maxiter::Int64 = 100)
    # Initial guess
    Ξ = Y' \ A'
    for i in 1:maxiter
        smallinds = abs.(Ξ) .<= ϵ
        Ξ[smallinds] .= 0.0
        for (j, y) in enumerate(eachrow(Y))
            biginds = vec(.!smallinds[j,:])
            Ξ[j, biginds] = A'[:, biginds] \ y
        end
    end
    return Ξ
end

function SInDy(X::AbstractArray, Ẋ::AbstractArray, Ψ::Basis; p::AbstractArray = [], ϵ::Number = 1e-1, maxiter::Int64 = 100)
    θ = hcat([Ψ(xi, p = p) for xi in eachcol(X)]...)
    Ξ = STRridge(θ, Ẋ, ϵ = ϵ, maxiter = maxiter)
    return simplify_constants.(Ξ*Ψ.basis)
end
