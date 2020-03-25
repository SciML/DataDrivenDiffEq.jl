mutable struct ADMM{U} <: AbstractOptimiser
    λ::U
    ρ::U
end

ADMM() = ADMM(0.1, 1.0)


function set_threshold!(opt::ADMM, threshold)
    opt.λ = threshold*opt.ρ
end

get_threshold(opt::ADMM) = opt.λ/opt.ρ

init(o::ADMM, A::AbstractArray, Y::AbstractArray) =  A \ Y
init!(X::AbstractArray, o::ADMM, A::AbstractArray, Y::AbstractArray) =  ldiv!(X, qr(A, Val(true)), Y)

#soft_thresholding(x::AbstractArray, t::T) where T <: Real = sign.(x) .* max.(abs.(x) .- t, zero(eltype(x)))

function fit!(X::AbstractArray, A::AbstractArray, Y::AbstractArray, opt::ADMM; maxiter::Int64 = 100)
    n, m = size(A)

    g = NormL1(get_threshold(opt))

    x̂ = deepcopy(X)
    ŷ = zero(X)

    P = I(m)/opt.ρ - (A' * pinv(opt.ρ*I(n) + A*A') *A)/opt.ρ
    c = P*(A'*Y)

    @inbounds for i in 1:maxiter
        x̂ .= P*(opt.ρ*X - ŷ) + c
        prox!(X, g, x̂ + ŷ/opt.ρ)
        ŷ .= ŷ + opt.ρ*(x̂ - X)
    end

    X[abs.(X) .< get_threshold(opt)] .= zero(eltype(X))
end
