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

abstract type AbstractAlgorithmCache <: AbstractDataDrivenResult end
"""
    AbstractDAGSRAlgorithm

Developer interface for differentiable directed-acyclic-graph symbolic-regression
algorithms. This interface is intended for solver extensions, not ordinary users.

# Interface

A subtype must provide an `options` field compatible with [`CommonAlgOptions`](@ref)
and a method `update_parameters!(cache::SearchCache{<:MyAlgorithm})`. The generic
cache initialization supplies the dataset, candidate population, and optimization
state. An algorithm that uses a different graph representation must also specialize
`init_model(alg, basis, dataset, intervals)`.

The generic `CommonSolve.solve!` path consumes the cache, repeatedly calls
`update_parameters!`, and returns a [`DataDrivenDiffEq.DataDrivenSolution`](@ref).
Custom algorithms should keep the `loss`, `keep`, and population semantics of
`CommonAlgOptions` or document any intentional differences.

# Example

```julia
struct MyDAGAlgorithm <: DataDrivenLux.AbstractDAGSRAlgorithm
    options::DataDrivenLux.CommonAlgOptions
end

DataDrivenLux.update_parameters!(cache::DataDrivenLux.SearchCache{<:MyDAGAlgorithm}) =
    nothing
```
"""
abstract type AbstractDAGSRAlgorithm <: AbstractDataDrivenAlgorithm end
@public AbstractDAGSRAlgorithm
abstract type AbstractSimplex end
abstract type AbstractErrorModel end
abstract type AbstractErrorDistribution end
abstract type AbstractConfigurationCache <: StatisticalModel end
abstract type AbstractRewardScale{risk} end

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
