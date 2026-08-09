"""
$(DocStringExtensions.README)
"""
module DataDrivenDiffEq

using LinearAlgebra: Diagonal, mul!, norm, normalize, svd

import DiffEqBase
using SciMLBase: SciMLBase, isdiscrete
using CommonSolve: CommonSolve, init
using Reexport: @reexport
import ModelingToolkitBase
using ModelingToolkitBase: AbstractSystem, MTKParameters, equations, get_iv, get_observed,
    independent_variable, observed, parameters, unknowns

using Parameters: @unpack, @with_kw
using Setfield: @set!

using SciMLStructures: SciMLStructures as SS
using SciMLPublic: @public
using SymbolicUtils: SymbolicUtils, arguments, iscall, issym, operation, simplify,
    symtype, term, unwrap
using Symbolics: Symbolics, @variables, Differential, Equation, Num, build_function,
    get_variables, value, variable, wrap

# Local Difference operator (removed from Symbolics v7)
include("./difference.jl")
export Difference

using Random: Random, shuffle
using QuadGK: quadgk
using Statistics: mean, median
using StatsBase: StatsBase, UnitRangeTransform, ZScoreTransform, summarystats
using StatsAPI: StatsAPI, StatisticalModel, bic, dof, fit, nobs, r2, rss

import DataInterpolations
using DataInterpolations: LinearInterpolation

using MLUtils: splitobs, DataLoader

using DocStringExtensions: DocStringExtensions, FIELDS, SIGNATURES, TYPEDEF
using RecipesBase: @recipe, @series

@reexport using CommonSolve: solve

@enum DDProbType begin
    Direct = 1 # Direct problem without further information
    Discrete = 2 # Time discrete problem
    Continuous = 3 # Time continuous problem
end

"""
    DDReturnCode

Return code for DataDrivenDiffEq solver results.

The values indicate successful convergence, generic failure, and termination due to
iteration, wall-time, absolute-tolerance, or relative-tolerance limits.
"""
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

"""
    AbstractBasis

Supertype for symbolic feature bases accepted by data-driven algorithms.

# Interface

Subtypes must provide symbolic equations, state and parameter accessors, and callable
in-place and out-of-place evaluation. Solver packages may use [`get_f`](@ref),
[`is_implicit`](@ref), and [`is_controlled`](@ref) to inspect these capabilities.
"""
abstract type AbstractBasis <: AbstractSystem end

# Collect the DataInterpolations Methods into an Interpolation Type
abstract type AbstractInterpolationMethod end
abstract type CollocationKernel end

# Algorithms
"""
    AbstractDataDrivenAlgorithm

Supertype for algorithms that solve data-driven problems.

# Interface

An algorithm package must define `CommonSolve.solve!` for
[`InternalDataDrivenProblem`](@ref). It may specialize [`get_fit_targets`](@ref) when its
regression targets differ from the problem's implicit data.
"""
abstract type AbstractDataDrivenAlgorithm end

"""
    AbstractDataDrivenResult

Supertype for algorithm-specific result objects stored by [`DataDrivenSolution`](@ref).

# Interface

Result types should implement the applicable `StatsAPI.StatisticalModel` accessors and
an `is_success(result)` predicate.
"""
abstract type AbstractDataDrivenResult <: StatisticalModel end

# Problem and solution
"""
    AbstractDataDrivenProblem{N, C, K}

Supertype for data containers consumed by data-driven algorithms.

# Interface

Problem subtypes must implement [`get_implicit_data`](@ref), [`get_oop_args`](@ref), and
[`remake_problem`](@ref). `N` is the numeric element type, `C` records whether controls
are present, and `K` records whether the problem is direct, discrete, or continuous.
"""
abstract type AbstractDataDrivenProblem{Number, Bool, DDProbType} end

# Define some alias type for easier dispatch
"""
    ABSTRACT_DIRECT_PROB{N, C}

Developer dispatch alias for direct [`AbstractDataDrivenProblem`](@ref) subtypes.
"""
const ABSTRACT_DIRECT_PROB{N, C} = AbstractDataDrivenProblem{N, C, DDProbType(1)}

"""
    ABSTRACT_DISCRETE_PROB{N, C}

Developer dispatch alias for discrete [`AbstractDataDrivenProblem`](@ref) subtypes.
"""
const ABSTRACT_DISCRETE_PROB{N, C} = AbstractDataDrivenProblem{N, C, DDProbType(2)}

"""
    ABSTRACT_CONT_PROB{N, C}

Developer dispatch alias for continuous [`AbstractDataDrivenProblem`](@ref) subtypes.
"""
const ABSTRACT_CONT_PROB{N, C} = AbstractDataDrivenProblem{N, C, DDProbType(3)}

abstract type AbstractDataDrivenSolution <: StatisticalModel end

# Fallback result and algorithm
struct ErrorDataDrivenResult <: AbstractDataDrivenResult end
struct ZeroDataDrivenAlgorithm <: AbstractDataDrivenAlgorithm end

## Basis

include("./basis/build_function.jl")
include("./basis/utils.jl")
include("./basis/type.jl")
export Basis
export jacobian, dynamics
export implicit_variables, states, controls
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

@public AbstractBasis, AbstractDataDrivenAlgorithm, AbstractDataDrivenResult,
    AbstractDataDrivenProblem, ABSTRACT_DIRECT_PROB, ABSTRACT_DISCRETE_PROB,
    ABSTRACT_CONT_PROB, InternalDataDrivenProblem, get_fit_targets, is_implicit,
    is_controlled, get_f, get_implicit_data, get_oop_args, remake_problem, assert_lhs,
    apply_transform, apply_transform!, __construct_basis

include("./utils/plot_recipes.jl")
include("./utils/build_basis.jl")

# Precompilation workload to improve startup time and TTFX
include("./precompilation.jl")

end # module
