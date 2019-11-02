# Simple ridge regression based upon the sindy-mpc
# repository, see https://arxiv.org/abs/1711.05501
# and https://github.com/eurika-kaiser/SINDY-MPC/blob/master/LICENSE
function STRridge(A::AbstractArray, Y::AbstractArray; ϵ::Number = 1e-3, maxiter::Int64 = 100)
    # Initial guess
    Ξ = A \ Y
    for i in 1:maxiter
        smallinds = abs.(Ξ) .<= ϵ
        Ξ[smallinds] .= 0.0
        for (j, y) in enumerate(eachcol(Y))
            biginds = @. ! smallinds[:, j]
            Ξ[biginds, j] = A[:, biginds] \ y
        end
    end
    return Ξ
end

# Returns a basis for the differential state
function SInDy(X::AbstractArray, Ẋ::AbstractArray, Ψ::Basis; p::AbstractArray = [], ϵ::Number = 1e-1, maxiter::Int64 = 100)
    θ = hcat([Ψ(xi, p = p) for xi in eachcol(X)]...)
    Ξ = STRridge(θ', Ẋ', ϵ = ϵ, maxiter = maxiter)
    return Basis(simplify_constants.(Ξ'*Ψ.basis), variables(Ψ), parameters = p)
end
