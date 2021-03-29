module DataDrivenDiffEq

using LinearAlgebra
using DiffEqBase
using ModelingToolkit

using QuadGK
using Statistics
#using DSP
using DataInterpolations

using Reexport
using Compat
using DocStringExtensions


@reexport using ModelingToolkit: states, parameters, independent_variable, observed, controls
@reexport using DataInterpolations: ConstantInterpolation, LinearInterpolation, QuadraticInterpolation, LagrangeInterpolation, QuadraticSpline, CubicSpline, BSplineInterpolation, BSplineApprox, Curvefit

using ModelingToolkit: AbstractSystem
# Basis and Koopman
abstract type AbstractBasis <: AbstractSystem end
abstract type AbstractKoopman <: AbstractBasis end
# Collect the DataInterpolations Methods into an Interpolation Type
abstract type AbstractInterpolationMethod end
abstract type CollocationKernel end

# Algortihms for Koopman
abstract type AbstractKoopmanAlgorithm end

# Problem and solution
abstract type AbstractDataDrivenProblem end
abstract type AbstractDataDrivenSolution end


## Basis

include("./basis.jl")
export Basis
export jacobian
export free_parameters

include("./utils/basis_generators.jl")
export chebyshev_basis, monomial_basis, polynomial_basis
export sin_basis, cos_basis, fourier_basis

include("./utils/collocation.jl")
export InterpolationMethod
export EpanechnikovKernel, UniformKernel, TriangularKernel,QuarticKernel
export TriweightKernel, TricubeKernel, GaussianKernel, CosineKernel
export LogisticKernel, SigmoidKernel, SilvermanKernel
export collocate_data

include("./utils/utils.jl")
export AIC, AICC, BIC
export optimal_shrinkage, optimal_shrinkage!
export burst_sampling, subsample

## Koopman

include("./koopman/type.jl")
export Koopman
export operator, generator
export is_stable, is_discrete, is_continuous
export modes, frequencies, outputmap, updatable
export update!

include("./koopman/algorithms.jl")
export DMDPINV, DMDSVD, TOTALDMD

## Problem and Solution
include("./problem.jl")
export DataDrivenProblem
export DiscreteDataDrivenProblem, ContinuousDataDrivenProblem
export has_timepoints, has_inputs, has_observations, has_derivatives
export is_valid

include("./optimizers/Optimize.jl")
export sparse_regression!
export set_threshold!, get_threshold
export STLSQ, ADMM, SR3
export ImplicitOptimizer
export SoftThreshold, HardThreshold, ClippedAbsoluteDeviation

include("./solution.jl")
export DataDrivenSolution

include("./solve/sindy.jl")


##

#include("./optimizers/Optimize.jl")
#using .Optimize
#
#export set_threshold!, set_threshold
#export STRRidge, ADMM, SR3
#
#export ADM


#abstract type AbstractKoopmanOperator <: Function end;
#include("./koopman/algorithms.jl")
#export DMDPINV, DMDSVD, TOTALDMD
#
#include("./koopman/koopman.jl")
#export eigen, eigvals, eigvecs
#export modes, frequencies
#export is_discrete, is_continuous
#export operator, generator
#export inputmap, outputmap, updatable, isstable
#
#include("./koopman/linearkoopman.jl")
#export LinearKoopman, update!
#
#include("./koopman/nonlinearkoopman.jl")
#export NonlinearKoopman, reduce_basis
#
#include("./koopman/exact_dmd.jl")
#export DMD, gDMD
#
#include("./koopman/dmdc.jl")
#export DMDc, gDMDc
#
#include("./koopman/extended_dmd.jl")
#export EDMD, gEDMD
#
#include("./sindy/results.jl")
#export SparseIdentificationResult
#export print_equations
#export get_coefficients, get_error, get_sparsity, get_aicc
#
#include("./sindy/sindy.jl")
#export SINDy
#export sparse_regression, sparse_regression!
#
#function SInDy(Y, X, basis; opt = STRRidge(), kwargs...)
#    @warn("SInDy has been deprecated. Use SINDy to recover the same functionality.")
#    SINDy(Y, X, basis, opt; kwargs...)
#end
#
#function ISInDy(Y, X, basis; opt = ADM(), kwargs...)
#    @warn("ISInDy has been deprecated. Use ISINDy to recover the same functionality.")
#    ISINDy(Y, X, basis, opt; kwargs...)
#end
#
#export SInDy, ISInDy
#
#include("./sindy/isindy.jl")
#export ISINDy

#include("./system_conversions.jl")

#include("./utils.jl")
#export AIC, AICC, BIC
#export optimal_shrinkage, optimal_shrinkage!
#export savitzky_golay
#export burst_sampling, subsample



end # module
