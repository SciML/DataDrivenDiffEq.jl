"""
$(DocStringExtensions.README)
"""
module DataDrivenDiffEq

using DocStringExtensions
using LinearAlgebra
using DiffEqBase
using ModelingToolkit

using QuadGK
using Statistics
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
export jacobian, dynamics
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

## Sparse Regression

include("./optimizers/Optimize.jl")
@reexport using DataDrivenDiffEq.Optimize: sparse_regression!
@reexport using DataDrivenDiffEq.Optimize: set_threshold!, get_threshold
@reexport using DataDrivenDiffEq.Optimize: STLSQ, ADMM, SR3
@reexport using DataDrivenDiffEq.Optimize: ImplicitOptimizer, ADM
@reexport using DataDrivenDiffEq.Optimize: SoftThreshold, HardThreshold, ClippedAbsoluteDeviation

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

include("./solution.jl")
export DataDrivenSolution
export result, parameters, parameter_map, metrics, algorithm, inputs
export output

include("./solve/sindy.jl")
include("./solve/koopman.jl")
export solve

end # module
