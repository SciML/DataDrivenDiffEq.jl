# # [Nonlinear Time Continuous System](@id nonlinear_continuos)
#
# Similarly, we can use the [Extended Dynamic Mode Decomposition](https://link.springer.com/article/10.1007/s00332-015-9258-5) via a nonlinear [`Basis`](@ref) of observables. Here, we will look at a rather [famous example](https://arxiv.org/pdf/1510.03007.pdf) with a finite dimensional solution.
    
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
    
@parameters t
@variables u[1:2](t)
Ψ = Basis([u; u[1]^2], u, independent_variable = t)
res = solve(prob, Ψ, DMDPINV(), digits = 1)
system = result(res)
#md println(res) # hide
#md println(system) # hide
#md println(parameters(res)) # hide

    
# The underlying dynamics have been recovered correctly by the algorithm!
# Similarly we could use sparse identification to solve the problem

sparse_res = solve(prob, Ψ, STLSQ(), digits = 1)
#md println(sparse_res)

# And the resulting system

sparse_system = result(sparse_res)
#md println(sparse_system)

# We can also directly look at the parameters of each result

parameter_map(res)

# Note that we are using [`parameter_map`](@ref) instead of just `parameters`, which returns 
# a vector suitable to use with `ModelingToolkit`.

parameter_map(sparse_res)

# To simulate the system, we create an `ODESystem` from the result

# Both results can be converted into an `ODESystem`

@named sys = ODESystem(
    equations(sparse_system), 
    get_iv(sparse_system),
    states(sparse_system), 
    parameters(sparse_system)
    );

    
x0 = [u[1] => u0[1], u[2] => u0[2]]
ps = parameter_map(sparse_res)
    
# And simulated using `OrdinaryDiffEq.jl` using the (known) initial conditions and the parameter mapping of the estimation.

ode_prob = ODEProblem(sys, x0, tspan, ps)
estimate = solve(ode_prob, Tsit5(), saveat = prob.t);

# And look at the result
#md plot(solution, color = :black)
#md plot!(estimate, color = :red, linestyle = :dash)

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

