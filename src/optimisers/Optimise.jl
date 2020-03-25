module Optimise

using LinearAlgebra
using ProximalOperators


abstract type AbstractOptimiser end;
abstract type AbstractSubspaceOptimiser end

include("./strridge.jl")
include("./admm.jl")
include("./sr3.jl")

#Nullspace for implicit sindy
include("./adm.jl")

export init, init!, fit!, set_threshold!, set_threshold
export STRRidge, ADMM, SR3
export ADM


end
