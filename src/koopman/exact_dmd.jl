function DMD(X::AbstractArray)
    return DMD(X[:, 1:end-1], X[:, 2:end])
end

function DMD(X::AbstractArray, Y::AbstractArray)
    @assert size(X)[2] .== size(Y)[2] "Provide consistent dimensions for data"
    @assert size(Y)[1] .<= size(Y)[2] "Provide consistent dimensions for data"

    # Best Frob norm approximator
    A = Y*pinv(X)

    return LinearKoopman(A, zero(eltype(A))*I(size(A,1)), Y*X', X*X', true)
end

function gDMD(X::AbstractArray, Y::AbstractArray)
    @assert size(X)[2] .== size(Y)[2] "Provide consistent dimensions for data"
    @assert size(Y)[1] .<= size(Y)[2] "Provide consistent dimensions for data"

    # Best Frob norm approximator
    A = Y*pinv(X)

    return LinearKoopman(A, zero(eltype(A))*I(size(A,1)), Y*X', X*X', false)
end

function gDMD(X::DataInterpolations.AbstractInterpolation, dt::T) where T <: Real
    itp = hcat(X.(0:dt:dt*size(X, 2))...)
    X̂ = itp[:,1:end-1]
    Ŷ = itp[:,2:end]

    A = Ŷ*pinv(X̂)
    λ, ϕ = eigen(A)
    ω = log.(λ)/dt
    A = ϕ*Diagonal(ω)*inv(ϕ)

    return LinearKoopman(A, zero(eltype(A))*I(size(A,1)), [], [], false)
end
