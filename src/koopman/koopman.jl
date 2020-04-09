mutable struct Koopman{T, FP, FB, FC} <: abstractKoopmanOperator

    A::AbstractArray{T,2} # Approximation of the operator, Transition

    ψ::FP # Map from states to observable
    B::FB # Input mapping
    C::FC # Output mapping

    λ::AbstractArray{Complex{T},1} # Eigenvalues (discrete time)
    ω::AbstractArray{Complex{T},1} # Frequencies
    ϕ::AbstractArray{Complex{T},2} # Modes

    # For online DMD
    Qₖ::AbstractArray{T,2}
    Pₖ::AbstractArray{T,2}

    # Information
    islifted::Bool
    iscontrolled::Bool
end


# Purely linear system
function Koopman(A::AbstractArray{T,2}; Q::AbstractArray{T,2} = Array{T}(undef, 0, 0), P::AbstractArray{T,2} = Array{T}(undef, 0, 0), dt::R = 0.0) where {T <: Real, R <: Real}
    # Eigen Decomposition for solution
    Λ, W = eigen(A)

    if dt > 0.0
        # Casting Complex enforces results
        ω = log.(Complex.(Λ)) / dt
    else
        ω = eltype(A)[]
    end

    ψ(u, p, t) = identity(u)
    C(u, p, t) = identity(u)
    B(u, p, t) = zeros(u)

    return Koopman(A, ψ, B, C, Complex.(Λ), Complex.(ω), Complex.(W), Q, P, false, false)
end

# Autonomos
function Koopman(A::AbstractArray{T,2}, ψ, C ; B = nothing, Q::AbstractArray{T,2} = Array{T}(undef, 0, 0), P::AbstractArray{T,2} = Array{T}(undef, 0, 0), dt::R = 0.0) where {T <: Real, R <: Real}
    # Eigen Decomposition for solution
    Λ, W = eigen(A)

    if dt > 0.0
        # Casting Complex enforces results
        ω = log.(Complex.(Λ)) / dt
    else
        ω = eltype(A)[]
    end

    b(u,p,t) = isnothing(B) ? zeros(u) : B(u,p,t)

    return Koopman(A, ψ, b, C ,Complex.(Λ), Complex.(ω), Complex.(W), Q, P, true, !isnothing(B))
end

function Koopman(A::AbstractArray{T,2}, ψ::AbstractArray{T,2}, C::AbstractArray{T,2} ; B = nothing, Q::AbstractArray{T,2} = Array{T}(undef, 0, 0), P::AbstractArray{T,2} = Array{T}(undef, 0, 0), dt::R = 0.0) where {T <: Real, R <: Real}
    psi(u,p,t) = ψ*u
    c(u,p,t) = C*u

    return Koopman(A, psi, c, B = B, Q = Q, P = P, dt = dt)
end


# Keep it simple
LinearAlgebra.eigen(m::Koopman) = m.λ, m.ϕ
LinearAlgebra.eigvals(m::Koopman) = m.λ
LinearAlgebra.eigvecs(m::Koopman) = m.ϕ
modes(m::Koopman) = eigvecs(m)
frequencies(m::Koopman) =  !isempty(m.ω) ? m.ω : error("No continouos frequencies available.")

inputmap(m::Koopman) = m.B
lifting(m::Koopman) = m.ψ
outputmap(m::Koopman) = m.C

isstable(m::Koopman) = !isempty(m.ω) ? all(real.(frequencies(m)) .<= 0.0) : all(abs.(eigvals(m)) .<= 1)
iscontinouos(m::Koopman) = !isempty(m.ω) ? true : false
isdiscrete(m::Koopman) = !iscontinouos(m)
islifted(m::Koopman) = m.islifted
iscontrolled(m::Koopman) = m.iscontrolled
isupdateable(m::Koopman) = !isempty(m.Qₖ) && !isempty(m.Pₖ)


function update!(m::Koopman, x::AbstractArray, y::AbstractArray; p::AbstractArray = [], dt::T = 0.0, threshold::Float64 = 1e-3) where T <: Real
    @assert dt >= zero(typeof(dt)) "Provide positive dt!"
    @assert isupdateable(m) "Koopman operator is not updateable!"

    Xnew = m.ψ(x, p, 0.0)
    Ynew = m.ψ(y, p, 0.0)

    # Check the error
    ϵ = norm(Ynew - m.A*Xnew, 2)

    if ϵ < threshold
        return
    end

    m.Qₖ += Ynew*Xnew'
    m.Pₖ += Xnew*Xnew'
    m.A = m.Qₖ*inv(m.Pₖ)
    m.λ, m.ϕ = eigen(m.A)

    # TODO Add update for output mapping

    if dt > 0.0 && iscontinouos(m)
        # Casting Complex enforces results
        m.ω = log.(Complex.(m.λ)) / dt
    end

    return
end

# Make it callable
# General case (everything is a function)
function (o::Koopman)(u, p, t; force_discrete::Bool = false) where T <: Real
    if iscontinouos(o) && !force_discrete
        A = real.(o.ϕ*Diagonal(o.ω)*inv(o.ϕ))
        return o.C(A*o.ψ(u,p,t), p, t)
    else
        return o.C(o.A*o.ψ(u,p,t), p, t)
    end
end

function (o::Koopman)(du, u, p, t; force_discrete::Bool = false) where T <: Real
    if iscontinouos(o) && !force_discrete
        A = real.(o.ϕ*Diagonal(o.ω)*inv(o.ϕ))
        du .= o.C(A*o.ψ(u,p,t), p, t)
    else
        du .=  o.C(o.A*o.ψ(u,p,t), p, t)
    end
    return
end

function dynamics(o::Koopman; force_discrete::Bool = false, force_continouos::Bool = false)
    if (isdiscrete(o) || force_discrete) && !force_continouos

        function f_oop(u, p,t)
            return o.C(o.A*o.ψ(u,p,t), p, t)
        end

        function f_iip(du, u, p, t)
            du .= o.C(o.A*o.ψ(u,p,t), p, t)
        end

        return f_oop, f_iip
    else
        @assert iscontinouos(o) "Koopman has no continouos representation!"
        A = real.(o.ϕ*Diagonal(o.ω)*inv(o.ϕ))

        function df_oop(u,p,t)
            return o.C(A*o.ψ(u,p,t), p, t)
        end
        function df_iip(du, u, p, t)
            du.= o.C(A*o.ψ(u,p,t), p, t)
        end

        return df_oop, df_iip
    end
end

function linear_dynamics(o::Koopman; force_discrete::Bool = false, force_continouos::Bool = false)
    if (isdiscrete(o) || force_discrete) && !force_continouos

        function f_oop(u, p,t)
            return o.A*u
        end

        function f_iip(du, u, p, t)
            mul!(du, o.A, u)
        end

        return f_oop, f_iip
    else
        @assert iscontinouos(o) "Koopman has no continouos representation!"
        A = real.(o.ϕ*Diagonal(o.ω)*inv(o.ϕ))

        function df_oop(u,p,t)
            return A*u
        end
        function df_iip(du, u, p, t)
            mul!(du, A, u)
        end

        return df_oop, df_iip
    end
end

function reduce_basis(o::Koopman; threshold = 1e-5)
    @assert isa(o.ψ, Basis) "Lifting has to be a Basis!"
    # Reduction for basis
    b = outputmap(o)(o.A, [], 0.0)
    inds = sum(abs, b, dims = 1) .> threshold
    reduced_b = Basis(o.ψ.basis[vec(inds)], variables(o.ψ), parameters = parameters(o.ψ))
    return reduced_b
end
