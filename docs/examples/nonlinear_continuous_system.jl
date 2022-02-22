# # [Nonlinear Time Continuous System](@id nonlinear_continuos)
#Similarly, we can use the [Extended Dynamic Mode Decomposition](https://link.springer.com/article/10.1007/s00332-015-9258-5) via a nonlinear [`Basis`](@ref) of observables. Here, we will look at a rather [famous example](https://arxiv.org/pdf/1510.03007.pdf) with a finite dimensional solution.
    
using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using OrdinaryDiffEq
#md using Plots

function slow_manifold(du, u, p, t)
    du[1] = p[1] * u[1]
    du[2] = p[2] * (u[2] - u[1]^2)
end

u0 = [3.0; -2.0]
tspan = (0.0, 5.0)
p = [-0.8; -0.7]

problem = ODEProblem(slow_manifold, u0, tspan, p)
solution = solve(problem, Tsit5(), saveat = 0.01)
#md plot(solution) 

# Since we are dealing with a continuous system in time, we define the associated [`DataDrivenProblem`](@ref) accordingly using the measured states `X`, their derivatives `DX` and the time `t`.
    
prob = ContinuousDataDrivenProblem(solution)

# Additionally, we need to define the [`Basis`](@ref) for our lifting, before we `solve` the problem in the lifted space.
    
@variables u[1:2]
Ψ = Basis([u; u[1]^2], u)
res = solve(prob, Ψ, DMDPINV(), digits = 1)
system = result(res)
#md println(res) # hide
#md println(system) # hide
#md println(parameters(res)) # hide

    
# The underlying dynamics have been recovered correctly by the algorithm!
# Similarly we could use sparse identification to solve the problem

sparse_res = solve(prob, Ψ, STLSQ(), digits = 1)
println(sparse_res)

# And the resulting system

sparse_system = result(sparse_res)
println(sparse_system)

# We can also directly look at the parameters of each result

parameter_map(res)

# Note that we are using [`parameter_map`](@ref) instead of just `parameters`, which returns 
# a vector suitable to use with `ModelingToolkit`.

parameter_map(sparse_res)

#md # ## [Copy-Pasteable Code](@id linear_discrete_copy_paste)
#md #
#md # ```julia
#md # @__CODE__
#md # ```

## Test #src
for r_ in [res, sparse_res] #src
    @test all(l2error(r_) .< 1e-5) #src
    @test all(aic(r_) .> 1e3) #src
    @test all(determination(r_) .>= 0.96) #src
end #src

