module DataDrivenSparse

import DataDrivenDiffEq
using DataDrivenDiffEq: AbstractDataDrivenAlgorithm, AbstractDataDrivenResult,
    DDReturnCode, DataDrivenCommonOptions, DataDrivenSolution, InternalDataDrivenProblem

using CommonSolve: CommonSolve
using DocStringExtensions: FIELDS, TYPEDEF, TYPEDFIELDS
using Parameters: @unpack
using StatsAPI: StatsAPI, StatisticalModel, aicc, coef, dof, nobs, r2, rss
using SciMLPublic: @public

using LinearAlgebra: I, cholesky, dot, norm
using Printf: @printf
using Statistics: mean

"""
    AbstractSparseRegressionAlgorithm

Abstract interface for sparse-regression algorithms used by `DataDrivenSparse`.

Concrete subtypes are callable algorithm objects that solve a matrix regression
problem for each target row. They are used directly as `solve(prob, basis, alg)`
optimizers and through [`SparseLinearSolver`](@ref).

# Interface

A subtype `Alg <: AbstractSparseRegressionAlgorithm` must provide:

  - `get_thresholds(alg)`: returns the scalar threshold or iterable threshold
    schedule evaluated by [`SparseLinearSolver`](@ref).
  - `alg(X, Y; options = DataDrivenCommonOptions(), kwargs...)`: returns
    `(coefficients, optimal_thresholds, optimal_iterations)` for feature matrix
    `X` and target matrix `Y`.

Algorithms that use [`SparseLinearSolver`](@ref) should also implement:

  - `init_cache(alg, A::AbstractMatrix, B::AbstractMatrix)`: constructs the
    per-target cache for the design matrix `A` and target row matrix `B`.
  - `step!(cache, threshold)`: performs one thresholded solver update.

# Arguments

  - `X::AbstractMatrix`: feature or basis-evaluation matrix.
  - `Y::AbstractMatrix`: target matrix whose rows are fit independently.

# Keywords

  - `options::DataDrivenCommonOptions`: convergence tolerances, iteration
    limits, and verbosity used by iterative sparse-regression solvers.
  - `kwargs...`: algorithm-specific options. Generic callers should not depend
    on any keyword that is not documented by the concrete algorithm.

# Returns

The generic sparse-regression call returns a tuple
`(coefficients, optimal_thresholds, optimal_iterations)`. `coefficients` is a
matrix with one row per target in `Y`; the other entries record the selected
threshold and iteration count for each target row.

# Examples

```julia
using DataDrivenSparse

X = [1.0 2.0 3.0; 1.0 4.0 9.0]
Y = [2.0 4.0 6.0]
alg = STLSQ([0.1])

coefficients, thresholds, iterations = alg(X, Y)
```
"""
abstract type AbstractSparseRegressionAlgorithm <: AbstractDataDrivenAlgorithm end
@public AbstractSparseRegressionAlgorithm

"""
    AbstractProximalOperator

Developer interface for thresholding operators used by sparse-regression
algorithms such as [`SR3`](@ref).

# Interface

A subtype must implement `operator(x, λ)` as a callable object that updates `x` in
place, `operator(y, x, λ)` as an out-of-place-buffer form, and
`active_set!(mask, operator, x, λ)` to identify the nonzero coefficients. The two
callable forms must preserve the shape and element type of the coefficient array.
Concrete operators may store additional thresholds, but those fields and their
defaults must be documented. The in-place form returns the modified `x`; the
buffer form writes `y` and returns it. `active_set!` returns the modified mask.
"""
abstract type AbstractProximalOperator end

@public AbstractProximalOperator

"""
    active_set!(mask, operator, x, lambda)

Update the Boolean active-set mask for a sparse-regression proximal operator.

This is a developer extension point for [`AbstractProximalOperator`](@ref).
`mask` and `x` must have the same shape, and an active entry indicates that the
corresponding coefficient survives thresholding.

# Arguments

- `mask`: Boolean array with the same shape as `x`.
- `operator::AbstractProximalOperator`: thresholding operator.
- `x`: coefficient array inspected by the operator.
- `lambda`: nonnegative threshold parameter.

# Returns

Return the modified `mask`.
"""
function active_set! end

@public active_set!

"""
    AbstractSparseRegressionCache

Developer interface for the mutable cache used by
[`SparseLinearSolver`](@ref). This type is intended for packages implementing a
new [`AbstractSparseRegressionAlgorithm`](@ref), not for constructing user
results directly.

# Interface

A cache subtype must provide mutable fields `Ã`, `B̃`, `X`, `X_prev`, and
`active_set`. `Ã` is the feature matrix, `B̃` is the target vector or matrix,
`X` is the current coefficient array, `X_prev` is the previous iterate, and
`active_set` has the same shape as `X`. The generic solver calls `step!(cache,
λ)`, copies a winning cache with `_set!`, and checks convergence with the
`abstol` and `reltol` values from [`DataDrivenCommonOptions`](@ref).

The cache must also support the `StatsAPI` methods `coef`, `rss`, `dof`, and
`nobs`; the default methods supplied here use the fields above. A custom cache
should preserve the coefficient shape and return a numeric residual from `rss`.
"""
abstract type AbstractSparseRegressionCache <: StatisticalModel end
@public AbstractSparseRegressionCache

function _set!(x::AbstractSparseRegressionCache, y::AbstractSparseRegressionCache)
    begin
        foreach(eachindex(x.X)) do i
            x.X[i] = y.X[i]
            x.X_prev[i] = y.X_prev[i]
            x.active_set[i] = y.active_set[i]
        end
        return
    end
end

_zero!(x::AbstractSparseRegressionCache) = begin
    x.X .= zero(eltype(x.X))
    return
end

function _is_converged(x::AbstractSparseRegressionCache, abstol, reltol)::Bool
    @unpack X, X_prev, active_set = x
    !(any(active_set)) && return true
    Δ = norm(X .- X_prev)
    Δ < abstol && return true
    δ = Δ / norm(X)
    δ < reltol && return true
    return false
end

# StatsAPI interface
StatsAPI.coef(x::AbstractSparseRegressionCache) = getfield(x, :X)

StatsAPI.rss(x::AbstractSparseRegressionCache) = begin
    @unpack Ã, X, B̃ = x
    sum(abs2, X * Ã .- B̃)
end

StatsAPI.dof(x::AbstractSparseRegressionCache) = begin
    @unpack active_set = x
    sum(active_set)
end

StatsAPI.nobs(x::AbstractSparseRegressionCache) = begin
    @unpack B̃ = x
    return prod(size(B̃))
end

function StatsAPI.loglikelihood(x::AbstractSparseRegressionCache)
    return begin
        -nobs(x) / 2 * log(rss(x) / nobs(x))
    end
end

function StatsAPI.nullloglikelihood(x::AbstractSparseRegressionCache)
    return begin
        @unpack B̃ = x
        -nobs(x) / 2 * log(mean(abs2, B̃ .- mean(vec(B̃))))
    end
end

StatsAPI.r2(x::AbstractSparseRegressionCache) = r2(x, :CoxSnell)

##

include("algorithms/proximals.jl")
export SoftThreshold, HardThreshold, ClippedAbsoluteDeviation

"""
    get_thresholds(alg::AbstractSparseRegressionAlgorithm)

Return the scalar threshold or ordered threshold schedule explored by
[`SparseLinearSolver`](@ref). A custom algorithm must return either a scalar or
an iterable that supports `minimum` and iteration.
"""
get_thresholds(x::AbstractSparseRegressionAlgorithm) = getfield(x, :thresholds)

"""
    get_relaxation(alg::AbstractSparseRegressionAlgorithm)

Return an optional relaxation parameter used by an algorithm. The default is
`nothing`; algorithms that expose relaxation should specialize this method and
document how it changes their update rule.
"""
get_relaxation(x::AbstractSparseRegressionAlgorithm) = nothing

"""
    get_proximal(alg::AbstractSparseRegressionAlgorithm)

Return the [`AbstractProximalOperator`](@ref) used by an algorithm. The default
is [`SoftThreshold`](@ref).
"""
get_proximal(x::AbstractSparseRegressionAlgorithm) = SoftThreshold()

include("solver.jl")
export SparseLinearSolver

"""
    init_cache(alg, A, B)

Construct the mutable [`AbstractSparseRegressionCache`](@ref) for a sparse
regression algorithm. `A` contains features by observation and `B` contains
targets by observation. Implement this method for a custom algorithm before
using it with [`SparseLinearSolver`](@ref).
"""
function init_cache end

"""
    step!(cache, lambda)

Perform one thresholded update of a sparse-regression cache in place. The
implementation must update `cache.X`, `cache.X_prev`, and `cache.active_set`
consistently and return the cache or `nothing`.
"""
function step! end

@public get_thresholds, get_relaxation, get_proximal, init_cache, step!

function (x::X where {X <: AbstractSparseRegressionAlgorithm})(
        X, Y;
        options::DataDrivenCommonOptions = DataDrivenCommonOptions(),
        kwargs...
    )
    solver = SparseLinearSolver(x, options = options)
    results = solver(X, Y) # Keep this here for now

    coeff_matrix = zeros(eltype(X), size(Y, 1), size(X, 1))
    optimal_thresholds = []
    optimal_iterations = Int[]

    foreach(enumerate(results)) do (i, res)
        coeff_matrix[i:i, :] .= coef(res[1])
        push!(optimal_thresholds, res[2])
        push!(optimal_iterations, res[3])
    end

    return coeff_matrix, optimal_thresholds, optimal_iterations
end

include("algorithms/STLSQ.jl")
export STLSQ

include("algorithms/ADMM.jl")
export ADMM

include("algorithms/SR3.jl")
export SR3

include("algorithms/WyNDA.jl")
export WyNDA

include("algorithms/Implicit.jl")
export ImplicitOptimizer

include("result.jl")
include("commonsolve.jl")

end # module
