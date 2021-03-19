module Optimize

using LinearAlgebra

using ProgressMeter
using DocStringExtensions

abstract type AbstractOptimizerHistory end;

abstract type AbstractProximalOperator end;

abstract type AbstractOptimizer{T} end;
abstract type AbstractSubspaceOptimizer{T} <: AbstractOptimizer{T} end;

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
init(X::AbstractArray, o::AbstractOptimizer, A::AbstractArray, Y::AbstractArray) =  ldiv!(X, qr(A, Val(true)), Y)

init(o::AbstractSubspaceOptimizer, A::AbstractArray, Y::AbstractArray) = zeros(eltype(A), 2*size(A,2), size(Y, 2))
init!(X::AbstractArray, o::AbstractSubspaceOptimizer, A::AbstractArray, Y::AbstractArray) =  zeros(eltype(A), 2*size(A,2), size(Y, 2))

"""
$(SIGNATURES)

Clips the solution by the given threshold `λ` and ceils the entries to the corresponding decimal.
"""
@inline function clip_by_threshold!(x::AbstractArray, λ::T) where T <: Real
    dplace = ceil(Int, -log10(λ))
    for i in eachindex(x)
        x[i] = abs(x[i]) < λ ? zero(eltype(x)) : round(x[i], digits = dplace)
    end
    return
end

# Used to assemble the nullspace for regression
@inline _assemble_ns(A::AbstractMatrix, b::AbstractVector) = [hcat(map(i->b[i].*A[i,:], 1:size(A,1))...)' A]
@inline _assemble_ns(A::AbstractMatrix, B::AbstractMatrix) = map(x->_assemble_ns(A, x), eachcol(B))


# Evaluate the results for pareto
# Evaluate F
# At least one result
@inline G(opt::AbstractOptimizer{T} where T) = @views f->f[1] == 0 ? Inf : norm(f, 2)
# At least two results ( implict )
@inline G(opt::AbstractSubspaceOptimizer{T} where T) = @views f->f[1] <= 1 ? Inf : norm(f)

# Evaluate the regression
@inline F(opt::AbstractOptimizer{T} where T) = @views (x, A, y)->[norm(x, 0); norm(y .- A*x, 2)]

function F(opt::AbstractSubspaceOptimizer{T} where T)
    # For all inputs
    @views function f(x,A,y)
        reg = _assemble_ns(A, y)
        return [norm(x,0); norm(map(i->norm(reg[i]*x[:, i], 2), 1:size(y,2)), 2)]
    end
    # Just ns and regression
    @views f(x, A) = [norm(x, 0); norm(A*x, 2)]
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

include("./history.jl")

include("./stlsq.jl")
include("./admm.jl")
include("./sr3.jl")

#Nullspace for implicit sindy
include("./adm.jl")


# Init the progressmeters
# For a general optimizer
default_prg_msg() = "Solving sparse regression..."

function init_progress(opt::AbstractOptimizer, X, A, Y, maxiter)
    Progress(
        maxiter*length(opt.λ), 1, default_prg_msg()
    )
end

function init_progress(opt::ADM, X, A, Y, maxiter)
    Progress(
        maxiter*length(opt.λ)*size(Y, 2), 1, default_prg_msg()
    )
end

include("./sparseregression.jl")
export sparse_regression!


export init, init!, fit!, set_threshold!, get_threshold
export STLSQ, ADMM, SR3
export ADM

end
