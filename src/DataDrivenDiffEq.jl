module DataDrivenDiffEq

using LinearAlgebra
using ModelingToolkit
using QuadGK
using Statistics
using DSP
using Compat
using Parameters
using ControlSystems
using Lasso
using Plots

abstract type abstractBasis end;
abstract type abstractKoopmanOperator end;

include("./optimisers/Optimise.jl")
using .Optimise
export set_threshold!, set_threshold
export STRRidge, ADMM, SR3
export ADM

include("./basis.jl")
export Basis
export variables, jacobian, dynamics
export free_parameters, parameters, variables

include("./exact_dmd.jl")
export ExactDMD
export eigen, eigvals, eigvecs
export modes, frequencies, isstable
export dynamics, update!

include("./extended_dmd.jl")
export ExtendedDMD
export dynamics, linear_dynamics
export reduce_basis, update!

include("./dmdc.jl")
export DMDc
export eigen, eigvals, eigvecs
export get_dynamics, get_input_map, dynamics

include("./sindy.jl")
export SInDy
export sparse_regression, sparse_regression!

include("./isindy.jl")
export ISInDy

include("./havok/DelayEmbedding.jl")
export DelayEmbedding, HankelMatrix, embed

include("./havok/NumericalDifferentiation.jl")
export NumericalDifferentiation

include("./havok/RegressionSolve.jl")
export RegressionSolve, SequentialLeastSquares

include("./havok/havok.jl")
export HAVOKanalysis, HAVOKmodel, HAVOKsim, fit
export timeseries, delay, rank, timestep, embedding
export modes, eigenvalues, eigenseries
export derivative, coefficients, fulltspan
export forcing, dynamics

include("./havok/Visualization.jl")
export plot3d, plot
export heatmap, havok_modes, linear_intermittently_forced_dynamics, eigenseries_dist, forcing_dist

include("./utils.jl")
export AIC, AICC, BIC
export hankel, optimal_shrinkage, optimal_shrinkage!
export savitzky_golay
export burst_sampling, subsample
export PearsonCorrelates

end
