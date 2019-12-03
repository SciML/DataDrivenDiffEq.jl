module DataDrivenDiffEq

using LinearAlgebra
using ModelingToolkit
using Compat
import Convex, GLPKMathProgInterface;

abstract type abstractBasis end;
abstract type abstractKoopmanOperator end;

include("./basis.jl")
export Basis
export free_parameters, parameters, variables
export jacobian, dynamics

include("./exact_dmd.jl")
export ExactDMD
export eigen, eigvals, eigvecs
export modes, frequencies, isstable
export dynamics, update!, free_parameters


include("./extended_dmd.jl")
export ExtendedDMD
export dynamics, linear_dynamics
export reduce_basis, update!, free_parameters

include("./sindy.jl")
export SInDy

include("./model_selection.jl")
export AIC

end # module
