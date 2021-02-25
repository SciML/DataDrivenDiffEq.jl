module DataDrivenDiffEq

using LinearAlgebra
using DiffEqBase
using ModelingToolkit
using QuadGK
using Statistics
using DSP
using FiniteDifferences, DataInterpolations
using Compat
using DocStringExtensions

abstract type AbstractKoopmanOperator <: Function end;

include("./optimizers/Optimize.jl")
using .Optimize

export set_threshold!, set_threshold
export STLSQ, ADMM, SR3
export ADM
export SoftThreshold, HardThreshold, ClippedAbsoluteDeviation

include("./basis.jl")
export Basis
export variables, jacobian, dynamics
export free_parameters

include("./koopman/algorithms.jl")
export DMDPINV, DMDSVD, TOTALDMD

include("./koopman/koopman.jl")
export eigen, eigvals, eigvecs
export modes, frequencies
export is_discrete, is_continuous
export operator, generator
export inputmap, outputmap, updatable, isstable

include("./koopman/linearkoopman.jl")
export LinearKoopman, update!

include("./koopman/nonlinearkoopman.jl")
export NonlinearKoopman, reduce_basis

include("./koopman/exact_dmd.jl")
export DMD, gDMD

include("./koopman/dmdc.jl")
export DMDc, gDMDc

include("./koopman/extended_dmd.jl")
export EDMD, gEDMD

include("./sindy/results.jl")
export SparseIdentificationResult
export print_equations
export get_coefficients, get_error, get_sparsity, get_aicc

include("./sindy/sindy.jl")
export SINDy
export sparse_regression, sparse_regression!

function STRRidge(args...)
    @deprecate("STRRidge has been deprecated. Use STLSQ to recover the same functionality")
    return STLSQ(args...)
end

function SInDy(Y, X, basis; opt = STRRidge(), kwargs...)
    @deprecate("SInDy has been deprecated. Use SINDy to recover the same functionality.")
    SINDy(Y, X, basis, opt; kwargs...)
end

function ISInDy(Y, X, basis; opt = ADM(), kwargs...)
    @deprecate("ISInDy has been deprecated. Use ISINDy to recover the same functionality.")
    ISINDy(Y, X, basis, opt; kwargs...)
end

export SInDy, ISInDy, STRRidge

include("./sindy/isindy.jl")
export ISINDy

include("./system_conversions.jl")

include("./utils.jl")
export AIC, AICC, BIC
export optimal_shrinkage, optimal_shrinkage!
export savitzky_golay
export burst_sampling, subsample


include("./basis_generators.jl")
export chebyshev_basis, monomial_basis, polynomial_basis
export sin_basis, cos_basis, fourier_basis

end # module
