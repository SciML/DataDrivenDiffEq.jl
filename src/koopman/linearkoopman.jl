"""
    LinearKoopman(K, B, Q, P, discrete)

An approximation of the Koopman operator which is linear in the states.

`K` is the array representing the operator, `B` is the (possible present) array
representing the influence of exogenous inputs on the evolution.

`Q` and `P` are matrices used for updating the operator with new measurements.
`discrete` indicates if the operator is discrete or continuous.

The Koopman operator is callable with the typical signature of `f(u,p,t)` and `f(du,u,p,t)`, respectively.

# Example

```julia
k = LinearKoopman([1.0 0; 0 0.3], [], [], [], true)

u = k([2.0; 0.5], nothing, nothing)
du = similar(u)
k(du, u, nothing, nothing)
```
"""
mutable struct LinearKoopman <: AbstractKoopmanOperator
    operator::AbstractArray
    input::AbstractArray

    Q::AbstractArray
    P::AbstractArray

    discrete::Bool
end

outputmap(k::LinearKoopman) = throw(AssertionError("Linear Koopman Operator has no output map."))

function (k::LinearKoopman)(u, p, t)
    return k.operator*u
end

function (k::LinearKoopman)(du, u, p, t)
    mul!(du, k.operator, u)
end


"""
    update!(k, X, Y; threshold = eps())

Update the Koopman `k` given new data `X` and `Y`. The operator is updated in place if
the L2 error of the prediction exceeds the `threshold`.

"""
function update!(k::LinearKoopman, X::AbstractArray, Y::AbstractArray; threshold::T = eps()) where {T <: Real}
    @assert updatable(k) "Linear Koopman is not updatable."

    ϵ = norm(Y-operator(k)*X, 2)

    if ϵ < threshold
        return
    end

    k.Q += Y*X'
    k.P += X*X'
    k.operator .= k.Q*inv(k.P)
    return
end
