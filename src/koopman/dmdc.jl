function DMDc(X::AbstractArray, U::AbstractArray; B::AbstractArray = [], alg::AbstractKoopmanAlgorithm = DMDPINV())
    return DMDc(X[:, 1:end-1], X[:, 2:end], U, B = B, alg = alg)
end

function DMDc(X::AbstractArray, Y::AbstractArray, U::AbstractArray; B::AbstractArray = [], alg::AbstractKoopmanAlgorithm = DMDPINV())
    @assert size(X)[2] .== size(Y)[2] "Provide consistent dimensions for data"
    @assert size(Y)[1] .<= size(Y)[2] "Provide consistent dimensions for data"
    @assert size(X)[2] == size(U)[2] "Provide consistent input data."

    nₓ = size(X)[1]
    nᵤ = size(U)[1]

    Ω = vcat(X, U)

    if isempty(B)
        G = alg(Ω, Y)

        A = G[:, 1:nₓ]
        B = G[:, nₓ+1:end]

    else
        A = alg(X, Y-B*U)
    end

    return LinearKoopman(A, B, Y*Ω', Ω*Ω', true)
end

function gDMDc(X::AbstractArray, Y::AbstractArray, U::AbstractArray; B::AbstractArray = [], alg::AbstractKoopmanAlgorithm = DMDPINV())
    @assert size(X)[2] .== size(Y)[2] "Provide consistent dimensions for data"
    @assert size(Y)[1] .<= size(Y)[2] "Provide consistent dimensions for data"
    @assert size(X)[2] == size(U)[2] "Provide consistent input data."

    nₓ = size(X)[1]
    nᵤ = size(U)[1]

    Ω = vcat(X, U)

    if isempty(B)
        G = alg(Ω, Y)

        A = G[:, 1:nₓ]
        B = G[:, nₓ+1:end]

    else
        A = alg(X, Y-B*U)
    end


    return LinearKoopman(A, B, Y*Ω', Ω*Ω', false)
end


function gDMDc(t::AbstractVector, X::AbstractArray, U::AbstractArray; dt::Real = 0.0, B::AbstractArray = [], alg::DataDrivenDiffEq.AbstractKoopmanAlgorithm = DMDPINV(), fdm::FiniteDifferences.FiniteDifferenceMethod = backward_fdm(5, 1), itp = CubicSpline, itp_u = LinearInterpolation)
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

    Û = zeros(eltype(U), size(U, 1), length(t̂))

    for (i, ui) in enumerate(eachrow(U))
        uitp_ = itp_u(ui, t)
        Û[i, :] .= uitp_.(t̂)
    end

    return gDMDc(X̂, Y, Û; B = B, alg = alg)
end
