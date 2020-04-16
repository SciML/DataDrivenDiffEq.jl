mutable struct LinearKoopman <: AbstractKoopmanOperator
    operator::AbstractArray
    input::AbstractArray

    Q::AbstractArray
    P::AbstractArray

    discrete::Bool
end

outputmap(k::LinearKoopman) = AssertionError("Linear Koopman Operator has no output map.")

function (k::LinearKoopman)(u, p, t)
    return k.operator*u
end

function (k::LinearKoopman)(du, u, p, t)
    mul!(du, k.operator, u)
end

function update!(k::LinearKoopman, X::AbstractArray, Y::AbstractArray; threshold::T = eps()) where {T <: Real}
    @assert updateable(k) "Linear Koopman is not updateable."

    ϵ = norm(Y-operator(k)*X, 2)

    if ϵ < threshold
        return
    end

    k.Q += Y*X'
    k.P += X*X'
    k.operator .= k.Q*inv(k.P)
    return
end
