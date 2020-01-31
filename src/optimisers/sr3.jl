# Based upon alg 1 in
# A unified sparse optimization framework to learn parsimonious physics-informed models from data
# by K Champion et. al.

mutable struct SR3{U,V, T} <: AbstractOptimiser
    λ::U
    ν::V
    R::T
end

function SR3(λ = 1e-1, ν = 1.0)
    R = NormL1(λ*ν)
    return SR3(λ, ν, R)
end

function set_threshold!(opt::SR3, threshold)
    opt.λ = threshold^2/(2*opt.ν)
    return
end


init(o::SR3, A::AbstractArray, Y::AbstractArray) =  A \ Y


function fit!(X::AbstractArray, A::AbstractArray, Y::AbstractArray, opt::SR3; maxiter::Int64 = 10)

    n, m = size(A)
    W = zero(X)

    # Init matrices
    P = inv(A'*A+I(m)/(2*opt.ν))
    X̂ = A'*Y
    for i in 1:maxiter
        # Solve rigde regression
        X .= P*(X̂+W/(2*opt.ν))
        # Add proximal iteration
        prox!(W, opt.R, X)
    end

    # This is the effective threshold of the SR3 algorithm
    # See Unified Framework paper supplementary material S1
    η = sqrt(2*opt.λ*opt.ν)
    X[abs.(X) .< η] .= zero(eltype(X))
    return
end
