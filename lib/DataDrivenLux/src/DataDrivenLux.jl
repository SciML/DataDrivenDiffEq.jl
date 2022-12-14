module DataDrivenLux

using DataDrivenDiffEq

# Load specific (abstract) types
using DataDrivenDiffEq: AbstractBasis
using DataDrivenDiffEq: AbstractDataDrivenAlgorithm
using DataDrivenDiffEq: AbstractDataDrivenResult
using DataDrivenDiffEq: AbstractDataDrivenProblem
using DataDrivenDiffEq: DDReturnCode, ABSTRACT_CONT_PROB, ABSTRACT_DISCRETE_PROB
using DataDrivenDiffEq: InternalDataDrivenProblem
using DataDrivenDiffEq: is_implicit, is_controlled

using DataDrivenDiffEq.DocStringExtensions
using DataDrivenDiffEq.CommonSolve
using DataDrivenDiffEq.CommonSolve: solve!
using DataDrivenDiffEq.StatsBase
using DataDrivenDiffEq.Parameters
using DataDrivenDiffEq.Setfield

using Reexport
@reexport using Optim
using Lux

using InverseFunctions
using TransformVariables
using NNlib
using Distributions
using DistributionsAD


using ChainRulesCore
using ComponentArrays

using IntervalArithmetic
using Random
using Distributed
using ProgressMeter
using Logging
using AbstractDifferentiation, ForwardDiff
using Optimisers

abstract type AbstractAlgorithmCache <: AbstractDataDrivenResult end
abstract type AbstractDAGSRAlgorithm <: AbstractDataDrivenAlgorithm end
abstract type AbstractSimplex end
abstract type AbstractErrorModel end
abstract type AbstractErrorDistribution end
abstract type AbstractConfigurationCache <: StatsBase.StatisticalModel end

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
include("./lux/simplex.jl")
export Softmax, GumbelSoftmax

# Nodes and Layers
include("./lux/path_state.jl")
export PathState
include("./lux/node.jl")
export FunctionNode
include("./lux/layer.jl")
export FunctionLayer
include("./lux/graph.jl")
export LayeredDAG

include("caches/dataset.jl")
export Dataset

include("caches/candidate.jl")
export Candidate

#include("caches/cache.jl")
#export SearchCache
#
#include("algorithms/randomsearch.jl")
#export RandomSearch

#include("algorithms/reinforce.jl")
#export Reinforce
#
#include("solve.jl")

end # module DataDrivenLux
