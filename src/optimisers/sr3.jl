# Based upon alg 1 in
# A unified sparse optimization framework to learn parsimonious physics-informed models from data
# by K Champion et. al.

mutable struct SR3{U,T} <: AbstractOptimiser where {T <: ProximableFunction}
    λ::U
    ν::U
    R::T
end

Base.show(io::IO, x::SR3) = print(io, "SR3($(get_threshold(x)), $(x.R))")

@inline function Base.print(io::IO, x::SR3)
    show(io, x)
end


function SR3(λ = 1e-1, ν = 1.0)
    R = NormL1
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
    f = opt.R(get_threshold(opt))

    n, m = size(A)
    W = copy(X)

    # Init matrices
    P = inv(A'*A+I(m)/(opt.ν))
    X̂ = A'*Y
    for i in 1:maxiter
        # Solve rigde regression
        X .= P*(X̂+W/(opt.ν))
        # Add proximal iteration
        prox!(W, f, X)
    end

    X[abs.(X) .< get_threshold(opt)] .= zero(eltype(X))
    return
end
