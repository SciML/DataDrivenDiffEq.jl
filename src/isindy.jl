@inline function softtreshholding!(x::AbstractArray, λ::AbstractFloat)
    x .= sign.(x) .*max.(abs.(x) .- λ, 0.0)
end

function ADM(Y::AbstractArray, q_init::AbstractArray, λ::Float64, ϵ::Float64, maxiter::Int64)
    q = q_init
    x = Y*q
    for k in 1:maxiter
        q_old = q
        softtreshholding!(x, λ)
        mul!(q, Y', x/norm(Y'*x, 2))
        if norm(q_old-q, 2) < ϵ
            break
        end
        mul!(x, Y, q)
    end
    q[abs.(q) .< λ] .= zero(eltype(q))
    return q
end

# Initvary
function ADM(Y::AbstractArray, λ::T , ϵ::Float64, maxiter::Int64)  where T <: Real
    Q = zeros(size(Y)[1], size(Y)[2])
    for i in 1:size(Y)[2]
        Q[:, i] = ADM(Y, Y[i, :], λ, ϵ, maxiter)
    end

    # Filter results for information
    Q = Q[:,sum.(abs, eachcol(Q)) .> 0]
    # Just relations between at least 2
    Q = Q[:, norm.(eachcol(Q), 0) .> 1]
    return Q
end

# Pareto
function ADM(Y::AbstractArray, λ::AbstractArray, ϵ::Float64, maxiter::Int64)
    Q = zeros(eltype(Y), size(Y)[1], size(Y)[2]*length(λ))
    @inbounds for k in 1:size(Y)[2]
        @inbounds for i in 1:length(λ)
            Q[:, length(λ)*(k-1)+i] = ADM(Y, Y[k,:], λ[i], ϵ, maxiter)
        end
    end

    # Filter results for information
    Q = Q[:,sum.(abs, eachcol(Q)) .> 0]
    # Just relations
    Q = Q[:, norm.(eachcol(Q), 0) .> 1]
    return Q
end

function ISInDy(X::AbstractArray, Ẋ::AbstractArray, Ψ::Basis, λ::Number; p::AbstractArray = [], cost::Function = (x, θ)->sqrt(norm(x, 0)^2 + norm(θ*x, 2)^2), ϵ::Number = 1e-1, maxiter::Int64 = 5000)
    θ = hcat([Ψ(xi, p = p) for xi in eachcol(hcat([X; Ẋ]))]...)
    θ₀ = nullspace(θ', atol = Inf)
    Ξ = ADM(θ₀, λ, ϵ, maxiter)
    # Find the best
    # Closure
    return Ξ, θ
    cost_(x) =  cost(x, θ)
    sort!(Ξ, dims = 2, by = cost_)
    basis = [convert(Operation,simplify_constants(ξ*bi)) for (ξ, bi) in zip(Ξ[:, 1], Ψ.basis) if abs.(ξ) >= λ]
    return Basis(basis, variables(Ψ), parameters = p)
end

function ISInDy(X::AbstractArray, Ẋ::AbstractArray, Ψ::Basis, λ::AbstractArray; p::AbstractArray = [], cost::Function = (x, θ)->sqrt(norm(x, 0)^2 + norm(θ*x, 2)^2), ϵ::Number = 1e-1, maxiter::Int64 = 5000)
    θ = hcat([Ψ(xi, p = p) for xi in eachcol(hcat([X; Ẋ]))]...)
    θ₀ = nullspace(θ', atol = Inf)
    Ξ = ADM(θ₀, λ, ϵ, maxiter)
    # Find the best
    # Closure
    cost_(x) = cost(x, θ)
    sort!(Ξ, dims = 2, by = cost_)
    # TODO Returns zeros columns; Why?
    return Ξ, θ
    basis = [convert(Operation,simplify_constants(ξ*bi)) for (ξ, bi) in zip(Ξ[:, 1], Ψ.basis)]
    return Basis(basis, variables(Ψ), parameters = p)
end
