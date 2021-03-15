module Optimize

using LinearAlgebra
using DocStringExtensions

abstract type AbstractOptimizer end;
abstract type AbstractSubspaceOptimizer end;

abstract type AbstractProximalOperator end;

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

@inline function clip_by_threshold!(x::AbstractArray, λ::T) where T <: Real
    for i in eachindex(x)
        x[i] = abs(x[i]) < λ ? zero(eltype(x)) : x[i]
    end
    return
end

include("./proximals.jl")
export SoftThreshold, HardThreshold,ClippedAbsoluteDeviation

include("./stlsq.jl")
include("./admm.jl")
include("./sr3.jl")

#Nullspace for implicit sindy
include("./adm.jl")

export init, init!, fit!, set_threshold!, get_threshold
export STLSQ, ADMM, SR3
export ADM

end
