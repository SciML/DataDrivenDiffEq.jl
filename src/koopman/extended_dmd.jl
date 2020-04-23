
function EDMD(X::AbstractArray, Ψ::AbstractBasis; p::AbstractArray = [], t::AbstractVector = [], C::AbstractArray = [], alg::AbstractKoopmanAlgorithm = DMDPINV())
    return EDMD(X[:, 1:end-1], X[:, 2:end], Ψ, p = p, t = t, C = C, alg = alg)
end

function EDMD(X::AbstractArray, Y::AbstractArray, Ψ::AbstractBasis; p::AbstractArray = [], t::AbstractVector = [], C::AbstractArray = [], alg::AbstractKoopmanAlgorithm = DMDPINV())
    @assert size(X)[2] .== size(Y)[2] "Provide consistent dimensions for data"
    @assert size(Y)[1] .<= size(Y)[2] "Provide consistent dimensions for data"

    # Based upon William et.al. , A Data-Driven Approximation of the Koopman operator

    # Number of states and measurements
    N,M = size(X)

    # Compute the transformed data
    Ψ₀ = Ψ(X, p, t)
    Ψ₁ = Ψ(Y, p, t)

    A = alg(Ψ₀, Ψ₁)

    # Transform back to states
    if isempty(C)
        C = X*pinv(Ψ₀)
    end

    # TODO Maybe reduce the observable space here
    return NonlinearKoopman(A, [], C , Ψ, Ψ₁*Ψ₀', Ψ₀*Ψ₀', true)
end


function gEDMD(X::AbstractArray, DX::AbstractArray, Ψ::AbstractBasis; p::AbstractArray = [], t::AbstractVector = [], C::AbstractArray = [], alg::AbstractKoopmanAlgorithm = DMDPINV())
    @assert size(X)[2] .== size(DX)[2] "Provide consistent dimensions for data"
    @assert size(DX)[1] .<= size(DX)[2] "Provide consistent dimensions for data"

    # Based upon William et.al. , A Data-Driven Approximation of the Koopman operator

    # Number of states and measurements
    N,M = size(X)

    # Compute the transformed data
    Ψ₀ = Ψ(X, p, t)
    Ψ₁ = Ψ(DX, p, t)

    A = alg(Ψ₀, Ψ₁)

    # Transform back to states
    if isempty(C)
        C = X*pinv(Ψ₀)
    end

    # TODO Maybe reduce the observable space here
    return NonlinearKoopman(A, [], C , Ψ, Ψ₁*Ψ₀', Ψ₀*Ψ₀', false)
end

function gEDMD(t::AbstractVector, X::AbstractArray, Ψ::DataDrivenDiffEq.AbstractBasis; dt::Real = 0.0, p::AbstractArray = [], C::AbstractArray = [], alg::DataDrivenDiffEq.AbstractKoopmanAlgorithm = DMDPINV(), fdm::FiniteDifferences.FiniteDifferenceMethod = backward_fdm(5, 1), itp = CubicSpline)
    @assert size(X, 2) == length(t) "Sample size must match."
    @assert test_comp = begin

        if itp ∈ [LinearInterpolation, QuadraticInterpolation, QuadraticSpline] && !isa(fdm, FiniteDifferences.Backward{typeof(fdm.grid), typeof(fdm.coefs)})
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
    return gEDMD(X̂, Y, Ψ, alg = alg, p = p, t = collect(t̂), C = C)
end
