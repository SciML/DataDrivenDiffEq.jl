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

    return Koopman(Ã, Λ, ω, W, Y*X', X*X', :ExactDMD)
end
