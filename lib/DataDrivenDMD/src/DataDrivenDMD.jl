module DataDrivenDMD

import DataDrivenDiffEq
using DataDrivenDiffEq: ABSTRACT_CONT_PROB, ABSTRACT_DISCRETE_PROB, AbstractBasis,
    AbstractDataDrivenAlgorithm, AbstractDataDrivenResult, Basis, DDReturnCode,
    DataDrivenSolution, InternalDataDrivenProblem, is_controlled, is_implicit, jacobian

using CommonSolve: CommonSolve, solve
using DocStringExtensions: FIELDS, SIGNATURES, TYPEDEF
using Parameters: @unpack
using SciMLPublic: @public
using Statistics: mean
using StatsAPI: StatsAPI, r2

using LinearAlgebra: Diagonal, Eigen, eigen, svd

const _EMPTY_MATRIX = Matrix(undef, 0, 0)

"""
    AbstractKoopmanAlgorithm

Developer interface for algorithms that estimate a Koopman operator or generator.
This interface is intended for DataDrivenDiffEq solver packages and advanced
extensions, not ordinary application code.

# Interface

A subtype must implement `alg(X, Y) -> (K, B)`, where `X` and `Y` are lifted data
matrices with one observation per column, `K` is an operator representation
convertible by `Matrix`, and `B` is the input map or an empty matrix when no
controls are used. A controlled implementation may additionally implement
`alg(X, Y, U) -> (K, B)`. The generic four-argument forms support a supplied
input map or `nothing` and are provided by this package.

To participate in the common `solve` workflow, the subtype must be usable by the
generic `DataDrivenDiffEq.get_fit_targets` and `CommonSolve.solve!` methods for
[`InternalDataDrivenProblem`](@ref). The two-argument method is required; the
three-argument method is required when the basis contains controls. The returned
`K` must represent a square operator on the lifted state space, and `B` must have
the corresponding output-by-control shape. A custom algorithm should preserve
these dimensions so that the result can be converted back to a
[`DataDrivenDiffEq.Basis`](@ref).

# Arguments

- `X::AbstractArray`: lifted input data, with features in rows and observations in
  columns.
- `Y::AbstractArray`: lifted target data with the same number of columns as `X`.
- `U::AbstractArray`: optional control data with one column per observation.
- `B::AbstractArray` or `nothing`: an optional input map supplied by the common
  four-argument adapter.

# Returns

Return `(K, B)`. `K` is an operator representation accepted by the result
constructor, and `B` is an input map or an empty matrix when the fit is
uncontrolled.

# Example

```julia
    using LinearAlgebra

struct MyKoopman <: DataDrivenDMD.AbstractKoopmanAlgorithm end

    function (::MyKoopman)(X, Y)
        return eigen(Y / X), zeros(eltype(X), size(Y, 1), 0)
    end
```
"""
abstract type AbstractKoopmanAlgorithm <: AbstractDataDrivenAlgorithm end

@public AbstractKoopmanAlgorithm

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
