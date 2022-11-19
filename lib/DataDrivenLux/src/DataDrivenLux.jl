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
using Zygote 
using Random

abstract type AbstractSimplex end

# Utilities
include("utilities.jl")

# Simplex
include("simplex.jl")
export Softmax, GumbelSoftmax

# Nodes and Layers
include("node.jl")
include("layer.jl")

# Algorithms


end # module DataDrivenLux
