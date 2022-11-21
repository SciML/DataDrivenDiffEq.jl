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

abstract type AbstractDAGSRAlgorithm <: AbstractDataDrivenAlgorithm end
abstract type AbstractSimplex end
abstract type AbstractErrorModel end
abstract type AbstractErrorDistribution end
abstract type AbstractConfigurationCache <: StatsBase.StatisticalModel end

## Utilities

"""
$(SIGNATURES)

Return the bounds of the `parameters` of a `Basis` as an `Interval`. 
If no bounds are given, returns ± ∞.
"""
function get_parameter_bounds(x::DataDrivenDiffEq.Basis)
    map(ModelingToolkit.parameters(x)) do p
        lower, upper = getbounds(p)
        Interval(lower, upper)
    end
end

"""
$(SIGNATURES)

Return the transformation of the `parameters` of a `Basis` as an `as`. 
If no bounds are given, returns ± ∞.
"""
function get_parameter_transformation(x::DataDrivenDiffEq.Basis)
    map(ModelingToolkit.parameters(x)) do p
        lower, upper = getbounds(p)
        as(SymbolicUtils.symtype(p), lower, upper)
    end
end


"""
$(SIGNATURES)

Return the distribution of the `parameters` of a `Basis`. 
If no distributions are given, returns Uniform(bounds(basis)).
"""
function get_parameter_distributions(x::DataDrivenDiffEq.Basis)
    map(ModelingToolkit.parameters(x)) do p
        if hasdist(p)
            return getdist(p)
        else
            lower, upper = getbounds(p)
            return Uniform(lower, upper)
        end
    end
end


## 
include("error_model.jl")
export AdditiveError, MultiplicativeError
export ObservedError

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


include("algorithms/dataset.jl")


include("configuration.jl")

export ConfigurationCache
export optimize_configuration!, evaluate_configuration!
export get_data_loglikelihood, get_configuration_dof, get_configuration_loglikelihood
export get_scales


include("algorithms/cache.jl")
export SearchCache
export update!

include("algorithms/randomsearch.jl")
export RandomSearch

include("solve.jl")

end # module DataDrivenLux
