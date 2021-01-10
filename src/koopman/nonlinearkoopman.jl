"""
    NonlinearKoopman(K, B, C, basis, Q, P, discrete)

An approximation of the Koopman operator which is nonlinear in the states.

`K` is the array representing the operator, `B` is the (possible present) array
representing the influence of exogenous inputs on the evolution.
`C` is the array mapping from the Koopman space to the original state space. `basis` is a
[Basis]@ref(Basis), mapping the state space to the Koopman space.

`Q` and `P` are matrices used for updating the operator with new measurements.
`discrete` indicates if the operator is discrete or continuous.

The Koopman operator is callable with the typical signature of `f(u,p,t)` and `f(du,u,p,t)`, respectively.

# Example

```julia
k = EDMD(X, basis)

u = k([2.0; 0.5], nothing, nothing)
du = similar(u)
k(du, u, nothing, nothing)
```
"""
mutable struct NonlinearKoopman <: AbstractKoopmanOperator
    operator::AbstractArray
    input::AbstractArray
    output::AbstractArray

    basis::Basis

    Q::AbstractArray
    P::AbstractArray

    discrete::Bool
end

function NonlinearKoopman(operator, input, output, basis, Q, P, discrete, lowrank::LRAOptions)
    op = LinearOperator(psvdfact(operator, lowrank))
    NonlinearKoopman(op, input, output, basis, Q, P, discrete)
end

function NonlinearKoopman(operator, input, output, basis, Q, P, discrete, lowrank::EmptyLRAOptions)
    NonlinearKoopman(operator, input, output, basis, Q, P, discrete)
end

(k::NonlinearKoopman)(u, p::DiffEqBase.NullParameters, t) = k(u, [], t)
(k::NonlinearKoopman)(du, u, p::DiffEqBase.NullParameters, t) = k(du, u, [], t)

function (k::NonlinearKoopman)(u,  p::AbstractArray = [], t = nothing)
    return k.output*k.operator*k.basis(u, p, t)
end

function (k::NonlinearKoopman)(du, u, p::AbstractArray = [], t = nothing)
    mul!(du, k.output*k.operator, k.basis(u, p, t))
end

"""
    update!(k, X, Y; p = [], t = [], threshold = eps())

Update the Koopman `k` given new data `X` and `Y`. The operator is updated in place if
the L2 error of the prediction exceeds the `threshold`.

`p` and `t` are the parameters of the basis and the vector of timepoints, if necessary.
"""
function update!(k::NonlinearKoopman, X::AbstractArray, Y::AbstractArray; p::AbstractArray = [], t::AbstractVector = [], threshold::T = eps()) where {T <: Real}
    @assert updatable(k) "Linear Koopman is not updatable."

    Ψ₀ = k.basis(X, p, t)
    Ψ₁ = k.basis(Y, p, t)

    ϵ = norm(Ψ₁-operator(k)*Ψ₀, 2)

    if ϵ < threshold
        return
    end

    k.Q += Ψ₁*Ψ₀'
    k.P += Ψ₀*Ψ₀'
    k.operator .= k.Q / k.P

    if norm(Y - outputmap(k)*k.operator*Ψ₀) < threshold
        return
    end

    # TODO Make this a proper rank 1 update
    k.output .= X / Ψ₀

    return
end

"""
    reduce_basis(k; threshold)

Reduces the `basis` of the nonlinear Koopman using the 1-norm of each row
of the matrix `C*K`. Rows where the threshold is not reached are deleted.
"""
function reduce_basis(k::NonlinearKoopman; threshold = 1e-5, kwargs...)
    b = k.output*k.operator
    inds = vec(sum(abs, b, dims = 1) .> threshold)
    return Basis(map(x->x.rhs, k.basis[inds]), variables(k.basis), parameters = parameters(k.basis), iv = independent_variable(k.basis), kwargs...)
end


"""
    ODESystem(k; threshold = eps())

Convert a `NonlinearKoopman` into an `ODESystem`. `threshold` determines the cutoff
for the entries of the matrix representing the state space evolution of the system.
"""
function ModelingToolkit.ODESystem(k::NonlinearKoopman; threshold = eps(), kwargs...)
    @assert threshold > zero(threshold) "Threshold must be greater than zero"

    eqs = Any[]
    A = outputmap(k)*k.operator
    A[abs.(A) .< threshold] .= zero(eltype(A))
    eqs = hcat([x.rhs for x in k.basis.eqs])
    b = Basis(simplify.(A*eqs)[:,1], variables(k.basis), parameters = parameters(k.basis), iv = independent_variable(k.basis))
    return ODESystem(b, kwargs...)
end
