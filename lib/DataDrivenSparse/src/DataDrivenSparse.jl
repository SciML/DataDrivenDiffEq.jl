module DataDrivenSparse

using Base: gettypeinfos
using CommonSolve: solve!
using DataDrivenDiffEq
# Load specific (abstract) types
using DataDrivenDiffEq: AbstractBasis
using DataDrivenDiffEq: AbstractDataDrivenAlgorithm
using DataDrivenDiffEq: AbstractDataDrivenResult
using DataDrivenDiffEq: AbstractDataDrivenProblem
using DataDrivenDiffEq: DDReturnCode, ABSTRACT_CONT_PROB, ABSTRACT_DISCRETE_PROB
using DataDrivenDiffEq: InternalDataDrivenProblem
using DataDrivenDiffEq: is_implicit, is_controlled

using DocStringExtensions

using Reexport
using CommonSolve
using StatsBase
using Parameters
using Setfield
using LinearAlgebra
using Printf

abstract type AbstractSparseRegressionAlgorithm <: AbstractDataDrivenAlgorithm end
abstract type AbstractProximalOperator end

abstract type AbstractSparseRegressionCache <: StatsBase.StatisticalModel end

_set!(x::AbstractSparseRegressionCache, y::AbstractSparseRegressionCache) = begin
    foreach(eachindex(x.X)) do i 
        x.X[i] = y.X[i]
        x.X_prev[i] = y.X_prev[i]
        x.active_set[i] = y.active_set[i]
    end
    return
end

_zero!(x::AbstractSparseRegressionCache) = begin
    x.X .= zero(eltype(x.X))
    return
end

function Base.show(io::IO, tup::Tuple{AbstractSparseRegressionCache, Int, Number})
    t, iterations, λ = tup
    @printf io "%14e %6d %6d   %14e   %14e\n" λ iterations dof(t) rss(t) aicc(t)
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



# StatsBase Overload
StatsBase.coef(x::AbstractSparseRegressionCache) = getfield(x, :X)

StatsBase.rss(x::AbstractSparseRegressionCache) = begin
    @unpack A, X, B = x
    sum(abs2, X*Matrix(A) .- Matrix(B))
end

StatsBase.dof(x::AbstractSparseRegressionCache) = begin
    @unpack active_set = x
    sum(active_set)
end

StatsBase.nobs(x::AbstractSparseRegressionCache) = begin
    @unpack B = x
    return length(B)
end

StatsBase.loglikelihood(x::AbstractSparseRegressionCache) = begin
    -nobs(x)/2*log(rss(x)/nobs(x))
end

StatsBase.nullloglikelihood(x::AbstractSparseRegressionCache) = begin
    @unpack B = x
    -nobs(x)/2*log(mean(abs2, Matrix(B) .- mean(Matrix(B))))
end

StatsBase.r2(x::AbstractSparseRegressionCache) = r2(x, :CoxSnell)

##

include("algorithms/proximals.jl")
export SoftThreshold, HardThreshold, ClippedAbsoluteDeviation

get_thresholds(x::AbstractSparseRegressionAlgorithm) = getfield(x, :thresholds)
get_relaxation(x::AbstractSparseRegressionAlgorithm) = nothing
get_proximal(x::AbstractSparseRegressionAlgorithm) = SoftThreshold()

include("solver.jl")
export SparseLinearSolver

include("algorithms/algorithms.jl")
export STLSQ, ADMM


end # module
