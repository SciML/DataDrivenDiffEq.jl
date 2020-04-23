mutable struct NonlinearKoopman <: AbstractKoopmanOperator
    operator::AbstractArray
    input::AbstractArray
    output::AbstractArray

    basis::AbstractBasis

    Q::AbstractArray
    P::AbstractArray

    discrete::Bool
end

outputmap(k::NonlinearKoopman) = k.output
inputmap(k::NonlinearKoopman) = k.input


(k::NonlinearKoopman)(u, p::DiffEqBase.NullParameters, t) = k(u, [], t)
(k::NonlinearKoopman)(du, u, p::DiffEqBase.NullParameters, t) = k(du, u, [], t)

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
    k.operator .= k.Q / k.P

    if norm(Y - outputmap(k)*k.operator*Ψ₀) < threshold
        return
    end

    # TODO Make this a proper rank 1 update
    k.output .= X / Ψ₀

    return
end

function reduce_basis(k::NonlinearKoopman; threshold = 1e-5)
    b = k.output*k.operator
    inds = vec(sum(abs, b, dims = 1) .> threshold)
    return Basis(k.basis[inds], variables(k.basis), parameters = parameters(k.basis), iv = independent_variable(k.basis))
end

function ModelingToolkit.ODESystem(k::NonlinearKoopman; threshold = eps())
    @assert threshold > zero(threshold) "Threshold must be greater than zero"

    eqs = Operation[]
    A = outputmap(k)*k.operator
    A[abs.(A) .< threshold] .= zero(eltype(A))
    @inbounds for i in 1:size(A, 1)
        eq = nothing
        for j in 1:size(A, 2)
            if !iszero(A[i,j])
                if isnothing(eq)
                    eq = A[i,j]*k.basis[j]
                else
                    eq += A[i,j]*k.basis[j]
                end
            end
        end
        isnothing(eq) ? nothing : push!(eqs, eq)
    end
    b = Basis(eqs, variables(k.basis), parameters = parameters(k.basis), iv = independent_variable(k.basis))
    return ODESystem(b)
end
