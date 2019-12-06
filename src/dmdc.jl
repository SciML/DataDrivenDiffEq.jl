mutable struct DMDc{K, B, Q, P} <: abstractKoopmanOperator
    koopman::K
    forcing::B

    Qₖ::Q # TODO implement for update
    Pₖ::P
end


function DMDc(X::AbstractArray, Y::AbstractArray, U::AbstractArray; B::AbstractArray = [], dt::Float64 = 0.0)
    @assert all(size(X) .== size(Y))
    @assert size(X)[2] .== size(U)[2]


    nₓ = size(X)[1]
    nᵤ = size(U)[1]

    if isempty(B)
        Ω = vcat(X, U)
        G = Y * pinv(Ω)

        Ã = G[:, 1:nₓ]
        B̃ = G[:, nₓ+1:end]
    else
        Ã = (Y - B*U)*pinv(X)
        B̃ = B
    end

    # Eigen Decomposition for solution
    Λ, W = eigen(Ã)

    # Right now now really useful, since we do not know
    # how to deal with B
    if dt > 0.0
        # Casting Complex enforces results
        ω = log.(Complex.(Λ)) / dt
    else
        ω = []
    end

    koopman = ExactDMD(Ã, Λ, ω, W, nothing, nothing)
    # TODO implement update method
    return DMDc(koopman, B̃, nothing, nothing)
end


function DMDc(X::AbstractArray, U::AbstractArray; B::AbstractArray = [], dt::Float64 = 0.0)
    @assert size(X)[2]-1 == size(U)[2] "Provide consistent input data."
    return DMDc(X[:, 1:end-1], X[:, 2:end], U, B = B, dt = dt)
end

# Some nice functions
LinearAlgebra.eigen(m::DMDc) = eigen(m.koopman)
LinearAlgebra.eigvals(m::DMDc) = eigvals(m.koopman)
LinearAlgebra.eigvecs(m::DMDc) = eigvecs(m.koopman)

DataDrivenDiffEq.isstable(m::DMDc) = isstable(m.koopman)

get_dynamics(m::DMDc) = m.koopman
get_input_map(m::DMDc) = m.forcing

# TODO this can be done better, maybe use macros
function dynamics(m::DMDc; control = nothing)
    control = isnothing(control) ? zero_callback(m) : control
    @inline function dudt_(u, p, t; y = control)
        m.koopman.Ã * u + m.forcing * y(u, p, t)
    end
    return dudt_
end


function zero_callback(m::DMDc)
    return length(size(m.forcing)) <= 1 ? (u, p, t) -> zero(eltype(m.forcing)) : (u, p, t) -> zeros(eltype(m.forcing), size(m.forcing)[2])
end
