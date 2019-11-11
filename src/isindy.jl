function findsparsenullspace(X::AbstractArray, λ::Number)
    Q = qr(X, Val(true))
    R = Q.R
    R[abs.(R) .<= λ] .= 0
    idx = findfirst([sum(oi) for oi in eachrow(R)] .≈ 0)
    if !isnothing(idx)
        idx += -1
        E = [-inv(R[1:idx-1, 1:idx-1])*R[1:idx-1,idx:end]; Diagonal(ones(size(R)[2]-idx+1))]
        Ξ = Matrix(Q.P*E)
        return Ξ
    end
    return []
end

# Pareto front
function ISInDy(X::AbstractArray, Ẋ::AbstractArray, b::Basis, λ₀::Number; p = [], maxiter::Int64 = 1)
    θ = vcat([b(xi, p = p) for xi in eachrow([X' Ẋ'])]'...)
    scores = []
    Ξ = reshape(Vector{Float64}(), size(b)[1], 0)
    for i in 1:max(maxiter, size(X)[1])
        𝛯 = findsparsenullspace(θ, λ₀*(1.2^(i-1)))
        for ξ in eachcol(𝛯)
            Ξᵢ = cat(Ξ, ξ, dims = 2)
            if rank(Ξᵢ) > rank(Ξ)
                Ξ = Ξᵢ
                push!(scores, norm(θ*Ξᵢ, 2))
            end
        end
    end
    return Ξ, scores
end
