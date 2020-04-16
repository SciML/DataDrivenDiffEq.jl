mutable struct NonlinearKoopman <: AbstractKoopmanOperator
    operator::AbstractArray
    input::AbstractArray
    output::AbstractArray

    basis::AbstractBasis

    Q::AbstractArray
    P::AbstractArray

    discrete::Bool
end

outputmap(k::NonlinearKoopman) = k.outputmap
inputmap(k::NonlinearKoopman) = k.inputmap

function (k::NonlinearKoopman)(u,  p::AbstractArray = [], t = nothing)
    return k.output*k.operator*k.basis(u, p, t)
end

function (k::NonlinearKoopman)(du, u, p::AbstractArray = [], t = nothing)
    mul!(du, k.output*k.operator, k.basis(u, p, t))
end

function update!(k::NonlinearKoopman, X::AbstractArray, Y::AbstractArray; p::AbstractArray = [], t::AbstractVector = [], threshold::T = eps()) where {T <: Real}
    @assert updateable(k) "Linear Koopman is not updateable."

    Ψ₀ = k.basis(X, p, t)
    Ψ₁ = k.basis(Y, p, t)

    ϵ = norm(Ψ₁-operator(k)*Ψ₀, 2)

    if ϵ < threshold
        return
    end

    k.Q += Ψ₁*Ψ₀'
    k.P += Ψ₀*Ψ₀'
    k.operator .= k.Q*inv(k.P)
    return
end

function reduce_basis(k::NonlinearKoopman; threshold = 1e-5)
    b = k.output*k.operator
    inds = vec(sum(abs, b, dims = 1) .> threshold)
    return Basis(k.basis[inds], variables(k.basis), parameters = parameters(k.basis), iv = independent_variable(k.basis))
end
