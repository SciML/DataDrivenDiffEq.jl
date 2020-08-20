module Optimize

using LinearAlgebra
using ProximalOperators
using Logging

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


include("./strridge.jl")
include("./admm.jl")
include("./sr3.jl")

#Nullspace for implicit sindy
include("./adm.jl")

export init, init!, fit!, set_threshold!, get_threshold
export STRRidge, ADMM, SR3
export ADM

end
