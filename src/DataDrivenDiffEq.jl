"""
$(DocStringExtensions.README)
"""
module DataDrivenDiffEq

using LinearAlgebra

using DiffEqBase
using CommonSolve
using Reexport

using Parameters
using Setfield

@reexport using ModelingToolkit
using ModelingToolkit: AbstractSystem, AbstractTimeDependentSystem
using SymbolicUtils: operation, arguments, iscall, issym
using Symbolics
using Symbolics: scalarize, variable, value
@reexport using ModelingToolkit: unknowns, parameters, independent_variable, observed,
                                 controls, get_iv, get_observed

using Random
using QuadGK
using Statistics
using StatsBase
@reexport using StatsBase: rss, r2, aic, aicc, bic, summarystats, loglikelihood,
                           nullloglikelihood, nobs, dof

using DataInterpolations
@reexport using DataInterpolations: ConstantInterpolation, LinearInterpolation,
                                    QuadraticInterpolation, LagrangeInterpolation,
                                    QuadraticSpline, CubicSpline, BSplineInterpolation,
                                    BSplineApprox, Curvefit

@reexport using MLUtils: splitobs, DataLoader
@reexport using StatsBase: ZScoreTransform, UnitRangeTransform

using DocStringExtensions
using RecipesBase

@reexport using CommonSolve: solve

@enum DDProbType begin
    Direct = 1 # Direct problem without further information
    Discrete = 2 # Time discrete problem
    Continuous = 3 # Time continuous problem
end

# We want to export the ReturnCodes

@enum DDReturnCode begin
    Success = 1
    Failed = 2
    ReachedMaxIters = 3
    ReachedTimeLimit = 4
    AbsTolLimit = 5
    RelTolLimit = 6
end

export DDReturnCode

# Helper
const __EMPTY_MATRIX = Matrix(undef, 0, 0)
const __EMPTY_VECTOR = Vector(undef, 0)

# Basis with an indicator for implicit use
abstract type AbstractDataDrivenFunction{Bool, Bool} end
abstract type AbstractBasis <: AbstractTimeDependentSystem end

# Collect the DataInterpolations Methods into an Interpolation Type
abstract type AbstractInterpolationMethod end
abstract type CollocationKernel end

# Algorithms
abstract type AbstractDataDrivenAlgorithm end
abstract type AbstractDataDrivenResult <: StatsBase.StatisticalModel end

# Problem and solution
abstract type AbstractDataDrivenProblem{Number, Bool, DDProbType} end

# Define some alias type for easier dispatch
const ABSTRACT_DIRECT_PROB{N, C} = AbstractDataDrivenProblem{N, C, DDProbType(1)}
const ABSTRACT_DISCRETE_PROB{N, C} = AbstractDataDrivenProblem{N, C, DDProbType(2)}
const ABSTRACT_CONT_PROB{N, C} = AbstractDataDrivenProblem{N, C, DDProbType(3)}

abstract type AbstractDataDrivenSolution <: StatsBase.StatisticalModel end

# Fallback result and algorithm
struct ErrorDataDrivenResult <: AbstractDataDrivenResult end
struct ZeroDataDrivenAlgorithm <: AbstractDataDrivenAlgorithm end

## Basis

include("./basis/build_function.jl")
include("./basis/utils.jl")
include("./basis/type.jl")
export Basis
export jacobian, dynamics
export implicit_variables, states
export get_parameter_values, get_parameter_map

include("./utils/basis_generators.jl")
export chebyshev_basis, monomial_basis, polynomial_basis
export sin_basis, cos_basis, fourier_basis

include("./utils/collocation.jl")
export InterpolationMethod
export EpanechnikovKernel, UniformKernel, TriangularKernel, QuarticKernel
export TriweightKernel, TricubeKernel, GaussianKernel, CosineKernel
export LogisticKernel, SigmoidKernel, SilvermanKernel
export collocate_data

include("./utils/utils.jl")
export optimal_shrinkage, optimal_shrinkage!

include("./problem/type.jl")

export DataDrivenProblem
export DiscreteDataDrivenProblem, ContinuousDataDrivenProblem, DirectDataDrivenProblem
export is_autonomous, is_discrete, is_direct, is_continuous, is_parametrized, has_timepoints
export is_valid, @is_applicable, get_name

include("./problem/set.jl")
export DataDrivenDataset
export DirectDataset, DiscreteDataset, ContinuousDataset

include("./utils/data_processing.jl")
export DataProcessing, DataNormalization

include("./utils/common_options.jl")
export DataDrivenCommonOptions

include("./commonsolve.jl")

include("./solution.jl")
export DataDrivenSolution
export get_algorithm, get_results, get_basis, is_converged, get_problem

include("./utils/plot_recipes.jl")
include("./utils/build_basis.jl")

end # module
