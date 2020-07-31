module Optimize

using LinearAlgebra
using ProximalOperators


abstract type AbstractOptimizer end;
abstract type AbstractSubspaceOptimizer end;
abstract type AbstractScalarizationMethod end;

include("./strridge.jl")
include("./admm.jl")
include("./sr3.jl")

#Nullspace for implicit sindy
include("./adm.jl")

export init, init!, fit!, set_threshold!, get_threshold
export STRRidge, ADMM, SR3
export ADM

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

#include("./pareto.jl")
#export ParetoCandidate
#export point, parameter, iter, threshold
#
#export WeightedSum, WeightedExponentialSum, GoalProgramming
#export weights
#
#export ParetoFront
#export assert_dominance, conditional_add!, set_candidate!

end
