module DataDrivenSparse

import DataDrivenDiffEq
using DataDrivenDiffEq: AbstractDataDrivenAlgorithm, AbstractDataDrivenResult,
    DDReturnCode, DataDrivenCommonOptions, DataDrivenSolution, InternalDataDrivenProblem

using CommonSolve: CommonSolve
using DocStringExtensions: FIELDS, TYPEDEF, TYPEDFIELDS
using Parameters: @unpack
using StatsAPI: StatsAPI, StatisticalModel, aicc, coef, dof, nobs, r2, rss

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
abstract type AbstractProximalOperator end

abstract type AbstractSparseRegressionCache <: StatisticalModel end

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

get_thresholds(x::AbstractSparseRegressionAlgorithm) = getfield(x, :thresholds)
get_relaxation(x::AbstractSparseRegressionAlgorithm) = nothing
get_proximal(x::AbstractSparseRegressionAlgorithm) = SoftThreshold()

include("solver.jl")
export SparseLinearSolver

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
