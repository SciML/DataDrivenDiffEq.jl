"""
    EDMD(X, basis; alg, p, t, C)
    EDMD(X, Y, basis; alg, p, t, C)

Approximates a 'NonlinearKoopman' with the `AbstractKoopmanAlgorithm` 'alg' from the data matrices `X` or `X` and `Y` respectively.
If only `X` is given, the data is split into `X[:, 1:end-1]` and `X[:, 2:end]`.

Additional keyworded arguments include `p` for the parameter of the basis and `t` for an array of time points.
`C` is the matrix representing the mapping from koopman space into state space.

# Example

```julia
@parameters p[1] t
@variables u[1:2](t)
h = Operation[u; sin.(u); cos(p[1]*t)]
basis = Basis(h, u, parameters = p, iv = t)
koopman = EDMD(X, basis, p = [2.0], t = collect(0:0.2:10.0), C = Float64[1 0 0 0 0; 0 1 0 0 0])
```
"""
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

    # The jacobian to get d/dt(Ψ) = d/dx(Ψ) dx/dt
    ∇ = jacobian(Ψ)

    Ψ₁ = hcat([∇(X[:,i], p, isempty(t) ? zero(eltype(X)) : t[i])*DX[:, i] for i in 1:size(X, 2)]...)
    #return Ψ₁

    A = alg(Ψ₀, Ψ₁)

    # Transform back to states
    if isempty(C)
        C = X*pinv(Ψ₀)
    end

    # TODO Maybe reduce the observable space here
    return NonlinearKoopman(A, [], C , Ψ, Ψ₁*Ψ₀', Ψ₀*Ψ₀', false)
end


"""
    gEDMD(X, Y, basis; alg, p, t, C)
    gEDMD(t, X, basis; dt, p, C, alg, fdm, itp)

Approximates a 'NonlinearKoopman' with the `AbstractKoopmanAlgorithm` 'alg' from the data matrices `X` and `Y`.
`X` should contain the state trajectory and `Y` the differential state trajectory.

If no measurements of the differential state is available, `gEDMD` can be called with measurement time points `t` as a first argument.
It will then create an interpolation using the interpolation method from `DataInterpolations.jl` defined in `itp`. The trajectory will then be resample
to equidistant measurements over time corresponding to the mean of `diff(t)` or `dt` if given.
The differential state measurements will be computed via 'FiniteDifferences.jl', given a `FiniteDifferenceMethod` in `fdm`.

# Example

```julia
koopman = gEDMD(X, Y, basis)

fdm = backward_fdm(5,1)
itp = CubicSpline
koopman = gEDMD(t, X, basis, fdm = fdm, itp = itp)
```
"""
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
