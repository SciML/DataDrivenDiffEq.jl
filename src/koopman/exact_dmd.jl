"""
    DMD(X; alg)
    DMD(X, Y; alg)

Approximates a 'LinearKoopman' with the `AbstractKoopmanAlgorithm` 'alg' from the data matrices `X` or `X` and `Y`, respectively.
If only `X` is given, the data is split into `X[:, 1:end-1]` and `X[:, 2:end]`.

# Example

```julia
alg = DMDPINV()
koopman = DMD(X, alg = alg)

koopman = DMD(X[:, 1:end-1], X[:, 2:end], alg = alg)
```
"""
function DMD(X::AbstractArray; alg::AbstractKoopmanAlgorithm = DMDPINV(), lowrank = EmptyLRAOptions())
    return DMD(X[:, 1:end-1], X[:, 2:end], alg = alg, lowrank = lowrank)
end

function DMD(X::AbstractArray, Y::AbstractArray; alg::AbstractKoopmanAlgorithm = DMDPINV(), lowrank = EmptyLRAOptions())
    @assert size(X)[2] .== size(Y)[2] "Provide consistent dimensions for data"

    # Best Frob norm approximator
    A = alg(X, Y)

    return LinearKoopman(A, zero(eltype(A))*I(size(A,1)), Y*X', X*X', true, lowrank)
end

"""
    gDMD(X, Y; alg)
    gDMD(t, X ; dt, alg, fdm, itp)

Approximates a 'LinearKoopman' with the `AbstractKoopmanAlgorithm` 'alg' from the data matrices `X` and `Y`.
`X` should contain the state trajectory and `Y` the differential state trajectory.

If no measurements of the differential state are available, `gDMD` can be called with measurement time points `t` as the first argument.
It will then create an interpolation using the interpolation method from `DataInterpolations.jl` defined in `itp`. The trajectory will then be resampled
to equidistant measurements over time corresponding to the mean of `diff(t)` or `dt`, if given.
The differential state measurements will be computed via 'FiniteDifferences.jl', given a `FiniteDifferenceMethod` in `fdm`.

# Example

```julia
koopman = gDMD(X, Y)

fdm = backward_fdm(5,1)
itp = CubicSpline
koopman = gDMD(t, X, fdm = fdm, itp = itp)
```
"""
function gDMD(X::AbstractArray, Y::AbstractArray; alg::AbstractKoopmanAlgorithm = DMDPINV(), lowrank = EmptyLRAOptions())
    @assert size(X)[2] .== size(Y)[2] "Provide consistent dimensions for data"

    # Best Frob norm approximator
    A = alg(X, Y)

    return LinearKoopman(A, zero(eltype(A))*I(size(A,1)), Y*X', X*X', false, lowrank)
end

function gDMD(t::AbstractVector, X::AbstractArray ; dt::Real = 0.0, alg::DataDrivenDiffEq.AbstractKoopmanAlgorithm = DMDPINV(), fdm::FiniteDifferences.FiniteDifferenceMethod = backward_fdm(5, 1), itp = CubicSpline, lowrank = EmptyLRAOptions())
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
        dx(t) = fdm(x->itp_(x), t)
        X̂[i, :] .= itp_.(t̂)
        Y[i, :] .= dx.(t̂)
    end
    return gDMD(X̂, Y, alg = alg, lowrank = lowrank)
end
