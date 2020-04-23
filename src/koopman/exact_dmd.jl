function DMD(X::AbstractArray; alg::AbstractKoopmanAlgorithm = DMDPINV())
    return DMD(X[:, 1:end-1], X[:, 2:end], alg = alg)
end

function DMD(X::AbstractArray, Y::AbstractArray; alg::AbstractKoopmanAlgorithm = DMDPINV())
    @assert size(X)[2] .== size(Y)[2] "Provide consistent dimensions for data"
    @assert size(Y)[1] .<= size(Y)[2] "Provide consistent dimensions for data"

    # Best Frob norm approximator
    A = alg(X, Y)

    return LinearKoopman(A, zero(eltype(A))*I(size(A,1)), Y*X', X*X', true)
end

function gDMD(X::AbstractArray, Y::AbstractArray; alg::AbstractKoopmanAlgorithm = DMDPINV())
    @assert size(X)[2] .== size(Y)[2] "Provide consistent dimensions for data"
    @assert size(Y)[1] .<= size(Y)[2] "Provide consistent dimensions for data"

    # Best Frob norm approximator
    A = alg(X, Y)

    return LinearKoopman(A, zero(eltype(A))*I(size(A,1)), Y*X', X*X', false)
end

function gDMD(t::AbstractVector, X::AbstractArray ; dt::Real = 0.0, alg::DataDrivenDiffEq.AbstractKoopmanAlgorithm = DMDPINV(), fdm::FiniteDifferences.FiniteDifferenceMethod = backward_fdm(5, 1), itp = CubicSpline)
    @assert size(X, 2) == length(t) "Sample size must match."
    @assert test_comp = begin

        if itp ∈ [LinearInterpolation, QuadraticInterpolation] && !isa(fdm, FiniteDifferences.Backward{typeof(fdm.grid), typeof(fdm.coefs)})
            false
        else
            true
        end
    end "LinearInterpolation and QuadraticInterpolation need to a backward finite difference method."

    Δt = iszero(dt) ? mean(diff(t)) : dt
    t̂ = t[1]:Δt:t[end]

    X̂ = zeros(eltype(X), size(X, 1), length(t̂))
    Y = similar(X̂)
    for (i, xi) in enumerate(eachrow(X))
        itp_ = itp(xi, t)
        dx(t) = FiniteDifferences.fdm(fdm, itp_, t)
        X̂[i, :] .= itp_.(t̂)
        Y[i, :] .= dx.(t̂)
    end
    return gDMD(X̂, Y, alg = alg)
end
