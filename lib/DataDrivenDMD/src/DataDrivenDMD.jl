module DataDrivenDMD

import DataDrivenDiffEq
using DataDrivenDiffEq: ABSTRACT_CONT_PROB, ABSTRACT_DISCRETE_PROB, AbstractBasis,
    AbstractDataDrivenAlgorithm, AbstractDataDrivenResult, Basis, DDReturnCode,
    DataDrivenSolution, InternalDataDrivenProblem, is_controlled, is_implicit, jacobian

using CommonSolve: CommonSolve, solve
using DocStringExtensions: FIELDS, SIGNATURES, TYPEDEF
using Parameters: @unpack
using Statistics: mean
using StatsAPI: StatsAPI, r2

using LinearAlgebra: Diagonal, Eigen, eigen, svd

const _EMPTY_MATRIX = Matrix(undef, 0, 0)

abstract type AbstractKoopmanAlgorithm <: AbstractDataDrivenAlgorithm end

# Results
include("./result.jl")
export KoopmanResult
export get_operator, get_inputmap, get_outputmap

# Algorithms
include("./algorithms.jl")
export DMDPINV
export DMDSVD
export TOTALDMD
export FBDMD

# Solve
include("./solve.jl")
export solve

end # module
