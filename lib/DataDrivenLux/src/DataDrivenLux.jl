module DataDrivenLux

import DataDrivenDiffEq
using DataDrivenDiffEq: AbstractDataDrivenAlgorithm,
    AbstractDataDrivenResult, AbstractDataDrivenProblem, Basis, DDReturnCode,
    DataDrivenCommonOptions, DataDrivenProblem, DataDrivenSolution,
    InternalDataDrivenProblem, get_parameter_values, implicit_variables, states

using DocStringExtensions: DocStringExtensions, FIELDS, TYPEDEF, SIGNATURES
using CommonSolve: CommonSolve
using ConcreteStructs: @concrete
using Setfield: @set!
using ModelingToolkitBase: equations, get_iv, get_variables, getbounds, getdefault, getdist,
    hasdefault, hasdist, observed, parameters, setdefault
using Symbolics: Num, substitute

using Optim: Optim, LBFGS
using Optimisers: Optimisers, Adam

using Lux: Lux, logsoftmax, softmax!
using LuxCore: LuxCore, AbstractLuxLayer, AbstractLuxWrapperLayer
using WeightInitializers: WeightInitializers, ones32, zeros32

using InverseFunctions: InverseFunctions, NoInverse
using TransformVariables: TransformVariables, as, transform_logdensity
using Distributions: Distributions, Distribution, Normal, Uniform, Univariate, dof,
    loglikelihood, logpdf, mean, mode, quantile, scale, truncated
using DistributionsAD: DistributionsAD
using StatsAPI: StatsAPI, StatisticalModel, aicc, nobs, nullloglikelihood, r2, rss

using ChainRulesCore: @ignore_derivatives
using ComponentArrays: ComponentArrays, ComponentVector

using IntervalArithmetic: IntervalArithmetic, Interval, interval
using ProgressMeter: ProgressMeter
using AbstractDifferentiation: AbstractDifferentiation
using ForwardDiff: ForwardDiff

using Logging: Logging, NullLogger, with_logger
using Random: Random, AbstractRNG
using Distributed: Distributed, pmap
using SciMLPublic: @public

const AD = AbstractDifferentiation

"""
    AbstractAlgorithmCache

Developer interface for the optimization state returned by a DataDrivenLux
algorithm. Concrete caches are [`SearchCache`](@ref) values and are stored in
the result of `solve`.
"""
abstract type AbstractAlgorithmCache <: AbstractDataDrivenResult end
"""
    AbstractDAGSRAlgorithm

Developer interface for differentiable directed-acyclic-graph symbolic-regression
algorithms. This interface is intended for solver extensions, not ordinary users.

# Interface

A subtype must provide an `options` field compatible with [`CommonAlgOptions`](@ref)
and methods `init_model(alg, basis, dataset, intervals)` and
`update_parameters!(cache::SearchCache{<:MyAlgorithm})`. The generic cache
initialization supplies the dataset, candidate population, and optimization state.
`init_model` must return a callable Lux model compatible with the basis and
dataset dimensions. `update_parameters!` must mutate `cache.p` or other algorithm
state in place and return `nothing`. An algorithm that uses the default layered
graph can reuse the generic `init_model` method.

The `init_model` method should retain the package's dispatch shape,
`(::MyAlgorithm, ::Basis, ::Dataset, intervals)`, so it is more specific than the
default method while remaining applicable to the common solver path.

The generic `CommonSolve.solve!` path consumes the cache, repeatedly calls
`update_parameters!`, and returns a [`DataDrivenDiffEq.DataDrivenSolution`](@ref).
Custom algorithms should keep the `loss`, `keep`, and population semantics of
`CommonAlgOptions` or document any intentional differences.

# Example

```julia
struct MyDAGAlgorithm <: DataDrivenLux.AbstractDAGSRAlgorithm
    options::DataDrivenLux.CommonAlgOptions
end

DataDrivenLux.init_model(alg, basis, dataset, intervals) =
    DataDrivenLux.LayeredDAG(
        length(basis), size(dataset.y, 1), 1, (1,), (identity,)
    )
DataDrivenLux.update_parameters!(cache::DataDrivenLux.SearchCache{<:MyDAGAlgorithm}) = nothing
```
"""
abstract type AbstractDAGSRAlgorithm <: AbstractDataDrivenAlgorithm end
@public AbstractDAGSRAlgorithm
"""
    AbstractSimplex

Developer interface for mappings used to normalize node weights onto the
probability simplex. A subtype is called as `simplex(rng, output, input, κ)`.
"""
>>>>>>> 70010c20 (docs: declare sublibrary developer APIs)
abstract type AbstractSimplex end

"""
    AbstractErrorModel

Developer interface for observation error models. A subtype is called as
`model(distribution, observation, prediction, scale)` and returns a log density.
"""
abstract type AbstractErrorModel end

abstract type AbstractErrorDistribution end
abstract type AbstractConfigurationCache <: StatisticalModel end

"""
    AbstractRewardScale{risk}

Developer interface for reward transformations used by search algorithms. A
subtype is called with a vector of losses and returns one reward per loss.
"""
abstract type AbstractRewardScale{risk} end

"""
    init_model(alg, basis, dataset, intervals)

Construct the callable Lux model used by a differentiable symbolic-regression
algorithm. `basis` supplies the feature count, `dataset` supplies target and
control dimensions, and `intervals` contains the interval-evaluated basis values
used to mask invalid inputs.

# Returns

Return a model accepted by `LuxCore.initialparameters`, `LuxCore.setup`, and the
call `(model)(inputs, parameters, state)`. A custom algorithm may specialize this
method when it does not use the default [`LayeredDAG`](@ref) representation.
"""
function init_model end

"""
    update_parameters!(cache)

Update the population parameters for a symbolic-regression search iteration.
The method is called by `update_cache!` after the retained candidates have been
selected. Mutate the cache in place and return `nothing`.
"""
function update_parameters! end

"""
    init_cache(alg::AbstractDAGSRAlgorithm, basis, problem; kwargs...)

Build the search cache consumed by the common `solve!` implementation. The
default method creates a [`Dataset`](@ref), calls [`init_model`](@ref), samples
the initial population, and initializes the optimizer state. A custom algorithm
may specialize this method when its cache representation differs from
[`SearchCache`](@ref).
"""
function init_cache end

"""
    convert_to_basis(candidate, parameters, options)

Convert the selected symbolic-regression candidate into a
[`DataDrivenDiffEq.Basis`](@ref). A custom graph implementation must provide this
method if it does not use the package's [`Candidate`](@ref) representation.
"""
function convert_to_basis end

@public init_model, init_cache, update_parameters!, convert_to_basis
@public AbstractAlgorithmCache, AbstractDAGSRAlgorithm, AbstractSimplex,
    AbstractErrorModel, AbstractRewardScale

@enum __PROCESSUSE begin
    SERIAL = 1
    THREADED = 2
    PARALLEL = 3
end

##
include("utils.jl")

##
include("custom_priors.jl")
export AdditiveError, MultiplicativeError
export ObservedModel

# Simplex
include("lux/simplex.jl")
export Softmax, GumbelSoftmax, DirectSimplex

# Nodes and Layers
include("lux/path_state.jl")
export PathState

include("lux/node.jl")
export FunctionNode

include("lux/layer.jl")
export FunctionLayer

include("lux/graph.jl")
export LayeredDAG

include("caches/dataset.jl")
export Dataset

include("caches/candidate.jl")
export Candidate

include("caches/cache.jl")
export SearchCache

include("algorithms/rewards.jl")
export RelativeReward, AbsoluteReward

include("algorithms/common.jl")

include("algorithms/randomsearch.jl")
export RandomSearch

include("algorithms/reinforce.jl")
export Reinforce

include("algorithms/crossentropy.jl")
export CrossEntropy

include("solve.jl")

end
