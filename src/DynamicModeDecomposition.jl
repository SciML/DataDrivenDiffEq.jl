module DynamicModeDecomposition

using LinearAlgebra
using ModelingToolkit

abstract type abstractKoopmanOperator end;

include("./exact_dmd.jl")
export ExactDMD
export eigen, eigvals, eigvecs
export modes, frequencies, isstable
export dynamics, update!

#include("./basis_functions.jl")
#export BasisFunction
#export BasisCandidate

include("./extended_dmd.jl")
export ExtendedDMD
export dynamics, linear_dynamics
# TODO check update
# export update!




end # module
