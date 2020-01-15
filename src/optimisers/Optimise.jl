module Optimise

using LinearAlgebra
using ProximalOperators


abstract type AbstractOptimiser end;

include("./strridge.jl")
include("./admm.jl")
include("./sr3.jl")

export init, fit!
export STRRidge, ADMM, SR3


end
