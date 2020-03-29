# Based upon alg 1 in
# A unified sparse optimization framework to learn parsimonious physics-informed models from data
# by K Champion et. al.
# and
#
# Code adapted from the implementation found at
# https://github.com/UW-AMO/sr3-matlab/blob/master/src/sr3.m
# by Travis Askham

mutable struct SR3{U,T} <: AbstractOptimiser
    λ::U
    ν::U
    R::T
end


Base.show(io::IO, x::SR3) = print(io, "SR3($(get_threshold(x)))")

@inline function Base.print(io::IO, x::SR3)
    show(io, x)
end


function SR3(λ = 1e-1, ν = 0.5, R = NormL0)
    return SR3(λ, ν, R)
end

function set_threshold!(opt::SR3, threshold)
    opt.λ = threshold
    return
end

get_threshold(opt::SR3) = opt.λ

init(o::SR3, A::AbstractArray, Y::AbstractArray) =  A \ Y
init!(X::AbstractArray, o::SR3, A::AbstractArray, Y::AbstractArray) =  ldiv!(X, qr(A, Val(true)), Y)

function fit!(X::AbstractArray, A::AbstractArray, Y::AbstractArray, opt::SR3; maxiter::Int64 = 10)
    f = opt.R(opt.λ)

    n, m = size(A)
    W = copy(X)

    # Init matrices
    P = qr(A'A+I(m)/opt.ν)
    X̂ = A'*Y
    for i in 1:maxiter
        # Solve rigde regression
        X .= P \ (X̂+W/opt.ν)
        # Add proximal iteration
        prox!(W, f, X, opt.λ*opt.ν)
    end
    return
end
