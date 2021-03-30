module Optimize

using LinearAlgebra
using Statistics

using ProgressMeter
using DocStringExtensions

abstract type AbstractOptimizerHistory end;

abstract type AbstractProximalOperator end;

abstract type AbstractOptimizer{T} end;
abstract type AbstractSubspaceOptimizer{T} <: AbstractOptimizer{T} end;

# Overload the norm
LinearAlgebra.norm(x, p, λ) = norm(x[abs.(x) .> λ], p)

# Dispatch on vector
function (opt::AbstractOptimizer{T} where T)(
        X::AbstractArray, A::AbstractArray, Y::AbstractVector{V} where V,
        args...; kwargs...
    )
    Y = reshape(Y, 1, length(Y))
    return opt(X, A, Y, args...; kwargs...)
end

function (opt::AbstractOptimizer{T} where T)(
        X::AbstractArray, A::AbstractArray, Y::Adjoint{V, AbstractVector{V}} where V,
        args...; kwargs...
    )
    Y = reshape(Y, 1, length(Y))
    return opt(X, A, Y, args...; kwargs...)
end

"""
$(SIGNATURES)

Set the threshold(s) of an optimizer to (a) specific value(s).
"""
function set_threshold!(opt::AbstractOptimizer{T}, threshold::T) where T <: AbstractVector
    @assert all(threshold .> zero(T)) "Threshold must be positive definite"
    opt.λ .= threshold
end

function set_threshold!(opt::AbstractOptimizer{T}, threshold::T) where T <: Number
    @assert all(threshold .> zero(T)) "Threshold must be positive definite"
    opt.λ = threshold
end

"""
$(SIGNATURES)

Get the threshold(s) of an optimizer.
"""
get_threshold(opt::AbstractOptimizer) = opt.λ

"""
$(SIGNATURES)

Initialize the optimizer with the least square solution for explicit and `zeros` for implicit optimization.
"""
init(o::AbstractOptimizer, A::AbstractArray, Y::AbstractArray) = A \ Y

init(o::AbstractSubspaceOptimizer, A::AbstractArray, Y::AbstractArray) = zeros(eltype(A), size(A, 2), size(Y, 2))

"""
$(SIGNATURES)

Initialize the optimizer with the least square solution for explicit and `zeros` for implicit optimization in place.
"""
init!(X::AbstractArray, o::AbstractOptimizer, A::AbstractArray, Y::AbstractArray) =  ldiv!(X, qr(A, Val(true)), Y)

"""
$(SIGNATURES)

Clips the solution by the given threshold `λ` and ceils the entries to the corresponding decimal.
"""
@inline function clip_by_threshold!(x::AbstractArray, λ::T, rounding::Bool = true) where T <: Real
    dplace = ceil(Int, -log10(λ))
    for i in eachindex(x)
        x[i] = abs(x[i]) < λ ? zero(eltype(x)) : x[i]
        x[i] = rounding ? round(x[i], digits = dplace) : x[i]
    end
    return
end


# Evaluate the results for pareto
G(opt::AbstractOptimizer{T} where T) = f->f[1] == 0 ? Inf : norm(f, 2)
G(opt::AbstractSubspaceOptimizer{T} where T) = f->f[1] <= 2 ? Inf : norm(f, 2)
# Evaluate F
function F(opt::AbstractOptimizer{T} where T)
    f(x, A, y::AbstractArray) = [norm(x, 0); norm(y .- A*x, 2)] # explicit
    f(x, A, y::AbstractArray, λ) = [norm(x, 0, λ); norm(y .- A*x, 2, λ)]
    f(x, A) = [norm(x,0); norm(A*x, 2)] # implicit
    f(x, A, λ::Number) = [norm(x,0,λ); norm(A*x, 2,λ)] # implicit
    return f
end

# Pareto
function evaluate_pareto!(current_parameter, tmp_parameter, fg::Function, args...)
    if fg(tmp_parameter, args...) < fg(current_parameter, args...)
        current_parameter .= tmp_parameter
        return true
    else
        return false
    end
end

include("./proximals.jl")
export SoftThreshold, HardThreshold,ClippedAbsoluteDeviation

# Remove the trace right now
#include("./history.jl")

include("./stlsq.jl")
include("./admm.jl")
include("./sr3.jl")

#Nullspace for implicit sindy
include("./adm.jl")
include("./implicit.jl")


# Init the progressmeters
# For a general optimizer
default_prg_msg(o::AbstractOptimizer) = summary(o)

function init_progress(opt::AbstractOptimizer, X, A, Y, maxiters, start)
    Progress(
        maxiters, default_prg_msg(opt), start
    )
end


include("./sparseregression.jl")
export sparse_regression!
export init, init!, set_threshold!, get_threshold
export STLSQ, ADMM, SR3
export ImplicitOptimizer, ADM

end
