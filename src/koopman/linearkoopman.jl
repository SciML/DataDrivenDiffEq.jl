"""
$(TYPEDEF)

An approximation of the Koopman operator which is linear in its states.
It is callable with the typical signature of `f(u,p,t)` and `f(du,u,p,t)`, respectively.


---

$(FIELDS)

---

# Example

```julia
k = LinearKoopman([1.0 0; 0 0.3], [], [], [], true)
u = k([2.0; 0.5], nothing, nothing)
du = similar(u)
k(du, u, nothing, nothing)
```

"""
mutable struct LinearKoopman <: AbstractKoopmanOperator
    "The operator or generator describing the dynamics"
    operator::Union{AbstractArray, LinearOperator}
    "Mapping of possible inputs onto the dynamics"
    input::AbstractArray
    "Used for internal rank-1 update"
    Q::AbstractArray
    "Used for internal rank-1 update"
    P::AbstractArray
    "Indicates if the system is discrete"
    discrete::Bool

    function LinearKoopman(operator, inputmap, Q, P, discrete)
        return new(operator, inputmap, Q, P, discrete)
    end
end


function LinearKoopman(operator, input, Q, P, discrete, lowrank::LRAOptions)
    op = LinearOperator(psvdfact(operator, lowrank))
    LinearKoopman(op, input, Q, P, discrete)
end

function LinearKoopman(operator, input, Q, P, discrete, lowrank::EmptyLRAOptions)
    LinearKoopman(operator, input, Q, P, discrete)
end

outputmap(k::LinearKoopman) = throw(AssertionError("Linear Koopman Operator has no output map."))

function (k::LinearKoopman)(u, p, t)
    return k.operator*u
end

function (k::LinearKoopman)(du, u, p, t)
    mul!(du, k.operator, u)
end


"""
$(TYPEDSIGNATURES)

Update the Koopman `k` given new data `X` and `Y`. The operator is updated in place if
the L2 error of the prediction exceeds the `threshold`.

`p` and `t` are the parameters of the basis and the vector of timepoints, if necessary.
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
