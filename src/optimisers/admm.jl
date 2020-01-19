mutable struct ADMM{U, T} <: AbstractOptimiser
    λ::U
    ρ::T
end

ADMM() = ADMM(0.1, 0.05)

init(o::ADMM, A::AbstractArray, Y::AbstractArray) =  A \ Y

#soft_thresholding(x::AbstractArray, t::T) where T <: Real = sign.(x) .* max.(abs.(x) .- t, zero(eltype(x)))

function fit!(X::AbstractArray, A::AbstractArray, Y::AbstractArray, opt::ADMM; maxiter::Int64 = 100)
    n, m = size(A)
    yn, ym = size(Y)
    @assert yn == n

    g = NormL1(opt.λ)

    x̂ = deepcopy(X)
    ŷ = zeros(eltype(Y), m, ym)

    P = I(m)/opt.ρ - (A' * pinv(opt.ρ*I(n) + A*A') *A)/opt.ρ
    c = P*(A'*Y)
    @inbounds for i in 1:maxiter
        x̂ .= P*(opt.ρ*X - ŷ) + c
        prox!(X, g, x̂ + ŷ/opt.ρ, opt.λ*opt.ρ)
        ŷ .= ŷ + opt.ρ*(x̂ - X)
    end

    X[abs.(X) .<= opt.λ] .= zero(eltype(X))
end
