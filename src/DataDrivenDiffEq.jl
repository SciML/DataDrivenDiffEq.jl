module DataDrivenDiffEq

using LinearAlgebra
using ModelingToolkit
using QuadGK
using Statistics
using DSP
using Compat

abstract type AbstractBasis end;
abstract type AbstractKoopmanOperator end;
abstract type AbstractDMDAlg end;

include("./optimisers/Optimise.jl")
using .Optimise
export set_threshold!, set_threshold
export STRRidge, ADMM, SR3
export ADM

include("./basis.jl")
export Basis
export variables, jacobian, dynamics
export free_parameters, parameters, variables

include("./koopman/dmd_algorithms.jl")
export DMDPINV

include("./koopman/koopman.jl")
export Koopman
export eigen, eigvals, eigvecs, modes, frequencies
export dynamics, linear_dynamics
export operator, generator, inputmap, lifting, outputmap
export update!
export isstable, iscontinouos, isdiscrete
export islifted, iscontrolled, isupdateable
export reduce_basis

include("./koopman/exact_dmd.jl")
export ExactDMD

include("./koopman/extended_dmd.jl")
export ExtendedDMD

include("./koopman/dmdc.jl")
export DMDc

include("./sindy.jl")
export SInDy
export sparse_regression, sparse_regression!

include("./isindy.jl")
export ISInDy

include("./utils.jl")
export AIC, AICC, BIC
export hankel, optimal_shrinkage, optimal_shrinkage!
export savitzky_golay
export burst_sampling, subsample

end # module
