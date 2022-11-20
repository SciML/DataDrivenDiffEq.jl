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
using DataDrivenDiffEq.StatsBase
using DataDrivenDiffEq.Parameters
using DataDrivenDiffEq.Setfield

using Reexport
@reexport using Optim
using Lux
using TransformVariables
using NNlib
using Distributions
using ChainRulesCore
using ComponentArrays
using Random

abstract type AbstractSimplex end

abstract type AbstractErrorModel end
abstract type AbstractErrorDistribution end
abstract type AbstractConfigurationCache <: StatsBase.StatisticalModel end

# Utilities
include("error_model.jl")
export AdditiveError, MultiplicativeError
export ObservedError

include("utilities.jl")



# Simplex
include("simplex.jl")
export Softmax, GumbelSoftmax

# Nodes and Layers
include("node.jl")
export DecisionNode
export update_state

include("layer.jl")
export DecisionLayer
export get_path
export LayeredDAG

include("configuration.jl")

export ConfigurationCache
export optimize_configuration!, evaluate!
export get_data_loglikelihood, get_configuration_dof, get_configuration_loglikelihood, 
export get_scales

end # module DataDrivenLux
