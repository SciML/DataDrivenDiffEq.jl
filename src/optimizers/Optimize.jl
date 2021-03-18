module Optimize

using LinearAlgebra

using ProgressMeter
using DocStringExtensions

abstract type AbstractOptimizerHistory end;

abstract type AbstractProximalOperator end;

abstract type AbstractOptimizer{T} end;
abstract type AbstractSubspaceOptimizer{T} <: AbstractOptimizer{T} end;


"""
$(SIGNATURES)

Set the threshold(s) of an optimizer to (a) specific value(s).
"""
function set_threshold!(opt::AbstractOptimizer{T}, threshold::T) where T <: Number
    @assert threshold > zero(T) "Threshold must be positive definite"
    opt.λ = threshold
end

function set_threshold!(opt::AbstractOptimizer{T}, threshold::T) where T <: AbstractVector
    @assert all(threshold .> zero(T)) "Threshold must be positive definite"
    opt.λ .= threshold
end

"""
$(SIGNATURES)

Get the threshold(s) of an optimizer.
"""
get_threshold(opt::AbstractOptimizer) = opt.λ

"""
$(SIGNATURES)

Initialize the optimizer with the least square solution.
"""
init(o::AbstractOptimizer, A::AbstractArray, Y::AbstractArray) = A \ Y
init(X::AbstractArray, o::AbstractOptimizer, A::AbstractArray, Y::AbstractArray) =  ldiv!(X, qr(A, Val(true)), Y)


@inline function clip_by_threshold!(x::AbstractArray, λ::T) where T <: Real
    # Also rounds

    for i in eachindex(x)
        x[i] = abs(x[i]) < λ ? zero(eltype(x)) : x[i]
    end
    return
end

include("./proximals.jl")
export SoftThreshold, HardThreshold,ClippedAbsoluteDeviation

include("./history.jl")

include("./stlsq.jl")
include("./admm.jl")
include("./sr3.jl")

#Nullspace for implicit sindy
#include("./adm.jl")

include("./sparseregression.jl")
export sparse_regression!


export init, init!, fit!, set_threshold!, get_threshold
export STLSQ, ADMM, SR3
export ADM

end
