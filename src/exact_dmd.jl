mutable struct ExactDMD{M,L,W,F, Q, P} <: abstractKoopmanOperator

    Ã::M # Approximation of the operator
    λ::L # Eigenvalues (discrete time)
    ω::W # Frequencies
    ϕ::F # Modes

    # For online DMD
    Qₖ::Q
    Pₖ::P

end

function ExactDMD(X::AbstractArray; Δt::Float64 = 0.0)
    return ExactDMD(X[:, 1:end-1], X[:, 2:end], Δt = Δt)
end

function ExactDMD(X::AbstractArray, Y::AbstractArray; Δt::Float64 = 0.0)
    @assert size(X)[2] .== size(Y)[2]
    @assert size(Y)[1] .<= size(Y)[2]

    # Best Frob norm approximator
    Ã = Y*pinv(X)
    # Eigen Decomposition for solution
    Λ, W = eigen(Ã)

    if Δt > 0.0
        # Casting Complex enforces results
        ω = log.(Complex.(Λ)) / Δt
    else
        ω = []
    end

    return ExactDMD(Ã, Λ, ω, W, Y*X', X*X')
end

# Keep it simple
eigen(m::ExactDMD) = m.λ, m.ϕ
eigvals(m::ExactDMD) = m.λ
eigvecs(m::ExactDMD) = m.ϕ

modes(m::ExactDMD) = eigvecs(m)
frequencies(m::ExactDMD) =  !isempty(m.ω) ? m.ω : error("No continouos frequencies available.")
isstable(m::ExactDMD) = !isempty(m.ω) ? all(real.(frequencies(m)) .<= 0.0) : all(abs.(eigvals(m)) .<= 1)

iscontinouos(m::ExactDMD) = !isempty(m.ω) ? true : false


# TODO this can be done better, maybe use macros
function dynamics(m::ExactDMD; discrete::Bool = true)
    if discrete
    # Return an inline function
        @inline function dudt_(du, u, p, t)
            mul!(du,m.Ã,u)
        end
        return dudt_

    else
        @assert iscontinouos(m)
        A = m.ϕ*Diagonal(m.ω)*inv(m.ϕ)

        @inline function dudt_c(du, u, p, t)
            mul!(du,A,u)
        end

        return dudt_c
    end
end


# Update with new measurements
function update!(m::ExactDMD, x::AbstractArray, y::AbstractArray; Δt::Float64 = 0.0, threshold::Float64 = 1e-3)
    # Check the error
    ϵ = norm(y - m.Ã*x, 2)

    if ϵ < threshold
        return
    end

    m.Qₖ += y*x'
    m.Pₖ += x*x'
    m.Ã = m.Qₖ*inv(m.Pₖ)
    m.λ, m.ϕ = eigen(m.Ã)

    if Δt > 0.0
        # Casting Complex enforces results
        ω = log.(Complex.(m.λ)) / Δt
    else
        ω = []
    end
    return
end
