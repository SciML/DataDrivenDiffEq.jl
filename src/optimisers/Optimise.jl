module Optimise

using LinearAlgebra
using ProximalOperators


abstract type AbstractOptimiser end;
abstract type AbstractSubspaceOptimiser end;
abstract type AbstractScalarizationMethod end;


include("./pareto.jl")
export ParetoCandidate
export point, parameter, iter, threshold

export WeightedSum, WeightedExponentialSum, GoalProgramming
export weights

export ParetoFront
export assert_dominance, conditional_add!, set_candidate!

include("./strridge.jl")
include("./admm.jl")
include("./sr3.jl")

#Nullspace for implicit sindy
include("./adm.jl")

export init, init!, fit!, set_threshold!, get_threshold
export STRRidge, ADMM, SR3
export ADM


end
