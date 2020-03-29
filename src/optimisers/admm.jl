# Adapted code for useage from
# https://web.stanford.edu/~boyd/papers/admm/basis_pursuit/basis_pursuit.html
# and the correlated paper
# Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers
# by Boyd et. al. 2010

mutable struct ADMM{U} <: AbstractOptimiser
    λ::U # threshold
    ρ::U # Lagrangian
    α::U # Relaxation
end

ADMM() = ADMM(0.1, 1.0 ,1.0)


function set_threshold!(opt::ADMM, threshold)
    opt.λ = threshold
end

get_threshold(opt::ADMM) = opt.λ

init(o::ADMM, A::AbstractArray, Y::AbstractArray) =  qr(A) \ Y
init!(X::AbstractArray, o::ADMM, A::AbstractArray, Y::AbstractArray) =  ldiv!(X, qr(A, Val(true)), Y)

#soft_thresholding(x::AbstractArray, t::T) where T <: Real = sign.(x) .* max.(abs.(x) .- t, zero(eltype(x)))

function fit!(X::AbstractArray, A::AbstractArray, Y::AbstractArray, opt::ADMM; maxiter::Int64 = 100)
    n, m = size(A)

    x̂ = zero(X)
    z = zero(X)
    u = zero(X)

    P = I(m) - A'*(A*A' \ A)
    Q = A'*(A*A' \ Y)


    @inbounds for i in 1:maxiter
        X .= P*(z-u) + Q
        x̂ .= opt.α*X+(one(opt.α)-opt.α)*z
        shrinkage!(z, x̂ + u, opt.λ\opt.ρ)
        u .+= x̂ - z
    end
end

function shrinkage!(z::AbstractArray, x::AbstractArray, κ::T) where T <: Number
    z .= sign.(x) .* max.(zero(x), abs.(x) .- κ)
end
