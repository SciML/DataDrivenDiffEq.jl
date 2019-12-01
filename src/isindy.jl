@inline function softtreshholding!(x::AbstractArray, λ::AbstractFloat)
    x .= sign.(x) .*max.(abs.(x) .- λ, 0.0)
end

function ADM(Y::AbstractArray, q_init::AbstractArray, λ::Float64, ϵ::Float64, maxiter::Int64)
    q = q_init
    x = Y'*q
    for k in 1:maxiter
        q_old = q
        softtreshholding!(x, λ)
        mul!(q, Y, x/norm(Y*x, 2))
        if norm(q_old-q, 2) < ϵ
            break
        end
        mul!(x, Y', q)
    end
    q[abs.(q) .< λ] .= zero(eltype(q))
    return q
end

# Initvary
function ADM(Y::AbstractArray, λ::T , ϵ::Float64, maxiter::Int64)  where T <: Real
    Q = zeros(size(Y)[1], size(Y)[2])
    for i in 1:size(Y)[2]
        Q[:, i] = ADM(Y, Y[:, i], λ, ϵ, maxiter)
    end

    # Filter results for information
    Q = Q[:,sum.(abs, eachcol(Q)) .> 0]
    # Just relations
    Q = Q[:, norm.(eachcol(Q), 0) .> 1]
    return Q
end

# Pareto
function ADM(Y::AbstractArray, λ::AbstractArray, ϵ::Float64, maxiter::Int64)
    Q = zeros(eltype(Y), size(Y)[1], size(Y)[2]*length(λ))
    @inbounds for k in 1:size(Y)[2]
        @inbounds for i in 1:length(λ)
            Q[:, length(λ)*(k-1)+i] = ADM(Y, Y[:,k], λ[i], ϵ, maxiter)
        end
    end

    # Filter results for information
    Q = Q[:,sum.(abs, eachcol(Q)) .> 0]
    # Just relations
    Q = Q[:, norm.(eachcol(Q), 0) .> 1]
    return Q
end
