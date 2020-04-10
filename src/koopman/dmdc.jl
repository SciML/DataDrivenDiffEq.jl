function DMDc(X::AbstractArray, U::AbstractArray; alg::O = DMDPINV(), B::AbstractArray = [], dt::T = 0.0, kwargs...) where {O <: AbstractDMDAlg, T <: Real}
    return DMDc(X[:, 1:end-1], X[:, 2:end], U, alg = alg, B = B, dt = dt, kwargs...)
end

function DMDc(X::AbstractArray, Y::AbstractArray, U::AbstractArray; alg::O = DMDPINV(), B::AbstractArray = [], dt::T = 0.0, kwargs...) where {O <: AbstractDMDAlg, T <: Real}
    @assert dt >= zero(dt)
    @assert size(X) == size(Y)
    @assert size(X)[2] .== size(U)[2]


    nₓ = size(X)[1]
    nᵤ = size(U)[1]

    if isempty(B)
        Ω = vcat(X, U)
        G = estimate_operator(alg, Ω, Y; kwargs...)

        A = G[:, 1:nₓ]
        B = G[:, nₓ+1:end]
    else
        A = estimate_operator(alg, X, Y - B*U; kwargs...)
    end

    # Create a function
    b(u, p, t) = length(size(B)) == 1 ? B.*u :  B*u
    ψ(u, p, t) = identity(u)
    C(u, p, t) = identity(u)

    return Koopman(A, ψ, C, B = b, Q = (Y-B*U)*X', P = X*X', dt = dt)
end
