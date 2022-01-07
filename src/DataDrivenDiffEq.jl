"""
$(DocStringExtensions.README)
"""
module DataDrivenDiffEq

using DocStringExtensions
using LinearAlgebra
using DiffEqBase
using ModelingToolkit

using Distributions
using QuadGK
using Statistics
using DataInterpolations
using ForwardDiff

using Requires
using ProgressMeter
using Reexport
using Compat
using DocStringExtensions


@reexport using ModelingToolkit: states, parameters, independent_variable, observed, controls, get_iv
@reexport using DataInterpolations: ConstantInterpolation, LinearInterpolation, QuadraticInterpolation, LagrangeInterpolation, QuadraticSpline, CubicSpline, BSplineInterpolation, BSplineApprox, Curvefit
using Symbolics: scalarize, variable


using ModelingToolkit: AbstractSystem
using ModelingToolkit: value, operation, arguments, istree, get_observed
# Basis and Koopman
abstract type AbstractBasis <: AbstractSystem end
abstract type AbstractKoopman <: AbstractBasis end
# Collect the DataInterpolations Methods into an Interpolation Type
abstract type AbstractInterpolationMethod end
abstract type CollocationKernel end

# Algortihms for Koopman
abstract type AbstractKoopmanAlgorithm end

# Abstract symbolic_regression
abstract type AbstractSymbolicRegression end

# Abstract Surrogate
abstract type AbstractSurrogate end

# Problem and solution
abstract type AbstractDataDrivenProblem{dType, cType, probType} end
abstract type AbstractDataDrivenSolution end




## Basis

include("./basis/build_function.jl")
include("./basis/utils.jl")
include("./basis/type.jl")
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
@reexport using DataDrivenDiffEq.Optimize: ImplicitOptimizer
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
export result, parameters, parameter_map, algorithm
export output, metrics, error, aic, determination

include("./solve/sindy.jl")
include("./solve/koopman.jl")
include("./solve/surrogates.jl")
export solve
export SurrogateSolvers

# Optional
function __init__()
    # Load and export OccamNet
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin

        using .Flux
        include("./symbolic_regression/occamnet.jl")

        export OccamNet,ProbabilityLayer
        export set_temp!
        export probability, logprobability
        export probabilities, logprobabilities
        export OccamSR

        @info "DataDrivenDiffEq : OccamNet is available."
    end

    @require SymbolicRegression = "8254be44-1295-4e6a-a16d-46603ac705cb" begin

        using .SymbolicRegression
        include("./symbolic_regression/symbolic_regression.jl")
        export EQSearch

        @info "DataDrivenDiffEq : Symboolic Regression is available."
    end

end

end # module
