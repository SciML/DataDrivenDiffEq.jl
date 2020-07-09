using DataDrivenDiffEq
using DataDrivenDiffEq.Optimize
using ModelingToolkit
using LinearAlgebra
@info "Loading OrdinaryDiffEq"
using OrdinaryDiffEq
@info "Loading DiffEqSensitivity"
using DiffEqSensitivity
@info "Loading Optim"
using Optim
@info "Loading DiffEqFlux"
using DiffEqFlux
@info "Loading Flux"
using Flux
using Test
@info "Finished loading packages"

include("./basis.jl")
include("./koopman.jl")
include("./sindy.jl")
include("./isindy.jl")
include("./utils.jl")
include("./optimize.jl")
include("./applications/partial_lotka_volterra.jl")
