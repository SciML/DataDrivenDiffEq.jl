module DataDrivenDiffEq

using LinearAlgebra
using ModelingToolkit
using Convex, GLPKMathProgInterface, Compat

abstract type abstractBasis end;
abstract type abstractKoopmanOperator end;

include("./basis.jl")
export Basis
export variables, jacobian, dynamics

include("./dynamicmodes/koopman.jl")
export Koopman
export eigen, eigvals, eigvecs
export modes, frequencies, operator, isstable
export dynamics, update!

include("./dynamicmodes/exact_dmd.jl")
export ExactDMD

include("./dynamicmodes/companion_dmd.jl")
export CompanionMatrixDMD

include("./dynamicmodes/hankel_dmd.jl")
export HankelDMD

include("./dynamicmodes/extended_dmd.jl")
export ExtendedDMD
export dynamics, linear_dynamics
export reduce_basis, update!

include("./sindy.jl")
export SInDy

end # module
