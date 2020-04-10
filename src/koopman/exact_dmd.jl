
function ExactDMD(X::AbstractArray; alg::O = DMDPINV(), dt::T = 0.0, kwargs...)  where {O <: AbstractDMDAlg, T <: Real}
    return ExactDMD(X[:, 1:end-1], X[:, 2:end], alg = alg, dt = dt, kwargs...)
end


function ExactDMD(X::AbstractArray, Y::AbstractArray; alg::O = DMDPINV(), dt::T = 0.0, kwargs...)  where {O <: AbstractDMDAlg, T <: Real}
    @assert dt >= zero(typeof(dt)) "Provide positive dt"
    @assert size(X) == size(Y) "Provide consistent dimensions for data"

    A = estimate_operator(alg, X, Y; kwargs...)

    return Koopman(A, Q = Y*X', P = X*X', dt = dt)
end
