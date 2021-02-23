module Optimize

using LinearAlgebra
using DocStringExtensions

abstract type AbstractOptimizer end;
abstract type AbstractSubspaceOptimizer end;

# Pareto
function evaluate_pareto!(current_parameter, tmp_parameter, fg::Function, args...)
    if fg(tmp_parameter, args...) < fg(current_parameter, args...)
        current_parameter .= tmp_parameter
        return true
    else
        return false
    end
end

# Pareto
export evaluate_pareto!

@inline function soft_thresholding!(x::AbstractArray, λ::T) where T <: Real
    for i in eachindex(x)
        x[i] = sign(x[i]) * max(abs(x[i]) - λ, zero(eltype(x)))
    end
    return
end

@inline function soft_thresholding!(y::AbstractArray, x::AbstractArray, λ::T) where T <: Real
    @assert all(size(y) .== size(x))
    for i in eachindex(x)
        y[i] = sign(x[i]) * max(abs(x[i]) - λ, zero(eltype(x)))
    end
    return
end

@inline function hard_thresholding!(x::AbstractArray, λ::T) where T <: Real
    for i in eachindex(x)
        x[i] = abs(x[i]) < λ ? zero(eltype(x)) : x[i]
    end
    return
end


include("./stlsq.jl")
include("./admm.jl")
include("./sr3.jl")

#Nullspace for implicit sindy
include("./adm.jl")

export init, init!, fit!, set_threshold!, get_threshold
export STLSQ, ADMM, SR3
export ADM

end
