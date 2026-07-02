module DataDrivenLux

using DataDrivenDiffEq

# Load specific (abstract) types
using DataDrivenDiffEq: AbstractDataDrivenAlgorithm,
    AbstractDataDrivenResult, AbstractDataDrivenProblem, DDReturnCode,
    InternalDataDrivenProblem

using DocStringExtensions: DocStringExtensions, FIELDS, TYPEDEF, SIGNATURES
using CommonSolve: CommonSolve
using ConcreteStructs: @concrete
using Setfield: Setfield, @set!

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
using StatsBase: StatsBase, aicc, nobs, nullloglikelihood, r2, rss, sum

using ChainRulesCore: @ignore_derivatives
using ComponentArrays: ComponentArrays, ComponentVector

using IntervalArithmetic: IntervalArithmetic, Interval, interval, isempty
using ProgressMeter: ProgressMeter
using AbstractDifferentiation: AbstractDifferentiation
using ForwardDiff: ForwardDiff

using Logging: Logging, NullLogger, with_logger
using Random: Random, AbstractRNG
using Distributed: Distributed, pmap

const AD = AbstractDifferentiation

abstract type AbstractAlgorithmCache <: AbstractDataDrivenResult end
abstract type AbstractDAGSRAlgorithm <: AbstractDataDrivenAlgorithm end
abstract type AbstractSimplex end
abstract type AbstractErrorModel end
abstract type AbstractErrorDistribution end
abstract type AbstractConfigurationCache <: StatsBase.StatisticalModel end
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
