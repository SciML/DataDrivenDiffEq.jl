"""
$(DocStringExtensions.README)
"""
module DataDrivenDiffEq

using DocStringExtensions
using LinearAlgebra
using DiffEqBase
using ModelingToolkit

using Flux
using Distributions

using QuadGK
using Statistics
using DataInterpolations

using Reexport
using Compat
using DocStringExtensions


@reexport using ModelingToolkit: states, parameters, independent_variable, observed, controls
@reexport using DataInterpolations: ConstantInterpolation, LinearInterpolation, QuadraticInterpolation, LagrangeInterpolation, QuadraticSpline, CubicSpline, BSplineInterpolation, BSplineApprox, Curvefit
using Symbolics: scalarize

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
abstract type AbstractDataDrivenProblem{dType, cType, probType} end
abstract type AbstractDataDrivenSolution end

# OccamNet
abstract type AbstractProbabilityLayer end

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

include("./symbolic_regression/occamnet.jl")
export OccamNet, set_temp!, probabilities, logprobabilities
@reexport using Flux: train!
@reexport using Flux: Descent, ADAM, Momentum, Nesterov, RMSProp,
	ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW,RADAM, OADAM, AdaBelief,
	InvDecay, ExpDecay, WeightDecay, stop, skip, Optimiser,
	ClipValue, ClipNorm

## Problem and Solution
# Use to distinguish the problem types
@enum DDProbType begin
    Direct=1 # Direct problem without further information
    Discrete=2 # Time discrete problem
    Continuous=3 # Time continous problem
end


# Define some alias type for easier dispatch
const AbstractDirectProb{N,C} = AbstractDataDrivenProblem{N,C,DDProbType(1)}
const AbstractDiscreteProb{N,C} = AbstractDataDrivenProblem{N,C,DDProbType(2)}
const AbstracContProb{N,C} = AbstractDataDrivenProblem{N,C,DDProbType(3)}


include("./problem.jl")

export DataDrivenProblem
export DiscreteDataDrivenProblem, ContinuousDataDrivenProblem, DirectDataDrivenProblem
export is_autonomous, is_discrete, is_direct, is_continuous, is_parametrized, has_timepoints
export is_valid


include("./solution.jl")
export DataDrivenSolution
export result, parameters, parameter_map, metrics, algorithm, inputs
export output

include("./solve/sindy.jl")
include("./solve/koopman.jl")
export solve

end # module
