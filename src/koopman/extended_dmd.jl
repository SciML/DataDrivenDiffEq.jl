
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
    return NonlinearKoopman(A, [], C , Ψ, Ψ₁*Ψ₀', Ψ₀*Ψ₀', false)
end
