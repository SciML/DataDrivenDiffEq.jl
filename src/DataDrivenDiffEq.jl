module DataDrivenDiffEq

using LinearAlgebra
using ModelingToolkit

abstract type abstractBasis end;
abstract type abstractKoopmanOperator end;

include("./basis.jl")
export Basis
export variables

include("./exact_dmd.jl")
export ExactDMD
export eigen, eigvals, eigvecs
export modes, frequencies, isstable
export dynamics, update!


include("./extended_dmd.jl")
export ExtendedDMD
export dynamics, linear_dynamics
# TODO check update
# export update!




end # module
