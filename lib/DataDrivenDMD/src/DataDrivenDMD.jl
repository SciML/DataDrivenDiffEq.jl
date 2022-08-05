module DataDrivenDMD

using CommonSolve: solve
using DataDrivenDiffEq
using DataDrivenDiffEq: AbstractBasis
using DataDrivenDiffEq: AbstractDataDrivenAlgorithm
using DataDrivenDiffEq: AbstractDataDrivenResult
using DataDrivenDiffEq: AbstractDataDrivenProblem

using ModelingToolkit

using DocStringExtensions

using LinearAlgebra
using CommonSolve
using StatsBase
using Parameters

using LinearSolve
using Symbolics

abstract type AbstractKoopman{J} <: AbstractBasis{J} end
abstract type AbstractKoopmanAlgorithm <: AbstractDataDrivenAlgorithm end

# The Koopman type
include("./type.jl")
export Koopman
export is_stable, is_discrete, is_continuous
export operator, generator
export frequencies, modes
export outputmap

# Results 
include("./result.jl")
export KoopmanResult

# Algorithms
include("./algorithms/DMDPINV.jl")
export DMDPINV

# Solve
include("./solve.jl")
export solve

end # module
