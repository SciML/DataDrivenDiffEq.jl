module DataDrivenDiffEq

using LinearAlgebra
using ModelingToolkit
using Convex, GLPKMathProgInterface, Compat

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

include("./sindy.jl")
export SInDy

include("./isindy.jl")
export ISInDy

end # module
