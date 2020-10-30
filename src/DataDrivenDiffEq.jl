module DataDrivenDiffEq

using LinearAlgebra
using ModelingToolkit
using ModelingToolkit: Term, Sym, FnType, value, <ₑ, isparameter
using Compat
import Convex, GLPKMathProgInterface;

abstract type abstractBasis end;
abstract type abstractKoopmanOperator end;

include("./basis.jl")
export Basis
export variables, jacobian, dynamics

include("./exact_dmd.jl")
export ExactDMD
export eigen, eigvals, eigvecs
export modes, frequencies, isstable
export dynamics, update!


include("./extended_dmd.jl")
export ExtendedDMD
export dynamics, linear_dynamics
export reduce_basis, update!

include("./dmdc.jl")
export DMDc
export eigen, eigvals, eigvecs
export get_dynamics, get_input_map, dynamics

include("./sindy.jl")
export SInDy

end # module
