module DataDrivenDMD

using CommonSolve: solve!
using DataDrivenDiffEq
# Load specific (abstract) types
using DataDrivenDiffEq: AbstractBasis
using DataDrivenDiffEq: AbstractDataDrivenAlgorithm
using DataDrivenDiffEq: AbstractDataDrivenResult
using DataDrivenDiffEq: AbstractDataDrivenProblem
using DataDrivenDiffEq: DDReturnCode, ABSTRACT_CONT_PROB, ABSTRACT_DISCRETE_PROB
using DataDrivenDiffEq: InternalDataDrivenProblem
using DataDrivenDiffEq: is_implicit, is_controlled

using DocStringExtensions

using LinearAlgebra
using CommonSolve
using StatsBase
using Parameters

abstract type AbstractKoopmanAlgorithm <: AbstractDataDrivenAlgorithm end

# Results 
include("./result.jl")
export KoopmanResult

# Algorithms
include("./algorithms.jl")
export DMDPINV
export DMDSVD
export TOTALDMD

# Solve
include("./solve.jl")
export solve

end # module
