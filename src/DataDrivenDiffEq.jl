"""
$(DocStringExtensions.README)
"""
module DataDrivenDiffEq

using LinearAlgebra
using StaticArrays

using DiffEqBase
using CommonSolve
using Reexport

using Parameters
using Setfield

using ModelingToolkit
using ModelingToolkit: AbstractSystem
using ModelingToolkit: value, operation, arguments, istree, get_observed
using ModelingToolkit.Symbolics
using ModelingToolkit.SymbolicUtils
using ModelingToolkit.Symbolics: scalarize, variable
@reexport using ModelingToolkit: states, parameters, independent_variable, observed,
                                 controls, get_iv

using Random
using QuadGK
using Statistics
using StatsBase
@reexport using StatsBase: rss, r2, aic, aicc, bic, summarystats, loglikelihood,
                           nullloglikelihood, nobs

using DataInterpolations
@reexport using DataInterpolations: ConstantInterpolation, LinearInterpolation,
                                    QuadraticInterpolation, LagrangeInterpolation,
                                    QuadraticSpline, CubicSpline, BSplineInterpolation,
                                    BSplineApprox, Curvefit

using DocStringExtensions
using RecipesBase

@reexport using CommonSolve: solve

@enum DDProbType begin
    Direct = 1 # Direct problem without further information
    Discrete = 2 # Time discrete problem
    Continuous = 3 # Time continous problem
end

# Basis with an indicator for implicit use
abstract type AbstractBasis{Bool} <: AbstractSystem end

# Collect the DataInterpolations Methods into an Interpolation Type
abstract type AbstractInterpolationMethod end
abstract type CollocationKernel end

# Algortihms
abstract type AbstractDataDrivenAlgorithm end
abstract type AbstractDataDrivenResult end

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

function CommonSolve.solve(::AbstractDataDrivenProblem, args...; kwargs...)
    @warn "No sufficient algorithm choosen!"
    return ErrorDataDrivenResult()
end

## Basis

include("./basis/build_function.jl")
include("./basis/utils.jl")
include("./basis/type.jl")
export Basis
export jacobian, dynamics
export implicit_variables

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

include("./problem/sample.jl")
export DataSampler, Split, Batcher

include("./solution.jl")
export DataDrivenSolution
export get_algorithm, get_result, get_basis, is_converged, get_problem

include("./utils/common_options.jl")
export DataDrivenCommonOptions

include("./utils/plot_recipes.jl")
include("./utils/build_basis.jl")


end # module
