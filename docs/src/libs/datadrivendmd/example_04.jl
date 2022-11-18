# # [Nonlinear Time Continuous System](@id nonlinear_continuos)
#
# Similarly, we can use the [Extended Dynamic Mode Decomposition](https://link.springer.com/article/10.1007/s00332-015-9258-5) via a nonlinear [`Basis`](@ref) of observables. Here, we will look at a rather [famous example](https://arxiv.org/pdf/1510.03007.pdf) with a finite dimensional solution.

using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenDMD
#md using Plots

function slow_manifold(du, u, p, t)
    du[1] = p[1] * u[1]
    du[2] = p[2] * (u[2] - u[1]^2)
end

u0 = [3.0; -2.0]
tspan = (0.0, 5.0)
p = [-0.8; -0.7]

problem = ODEProblem{true, SciMLBase.NoSpecialize}(slow_manifold, u0, tspan, p)
solution = solve(problem, Tsit5(), saveat = 0.1);
#md plot(solution) 

# Since we are dealing with a continuous system in time, we define the associated [`DataDrivenProblem`](@ref) accordingly using the measured states `X`, their derivatives `DX` and the time `t`.

prob = DataDrivenProblem(solution)
#md plot(prob)

# Additionally, we need to define the [`Basis`](@ref) for our lifting, before we `solve` the problem in the lifted space.

@parameters t
@variables u(t)[1:2]
Ψ = Basis([u; u[1]^2], u, independent_variable = t)
res = solve(prob, Ψ, DMDPINV(), digits = 2)
#md println(res) #hide

# We can also use different metrics on the `DataDrivenSolution` like the `aic`

#md aic(res)

# The `aicc`

#md aicc(res)

# The `bic`

#md bic(res)

# The `loglikelihood`

#md loglikelihood(res)

# And the number of parameters

#md dof(res)

# Lets have a closer look at the `Basis`

#md basis = get_basis(res) 
#md println(basis) #hide

# And the connected parameters

#md get_parameter_map(basis)

# And plot the results
#md plot(res)

#md # ## [Copy-Pasteable Code](@id linear_discrete_copy_paste)
#md #
#md # ```julia
#md # @__CODE__
#md # ```

## Test #src
for r_ in [res, sparse_res] #src
    @test all(l2error(r_) .< 1e-5) #src
    @test all(aic(r_) .< -1e3) #src
    @test all(determination(r_) .>= 0.96) #src
end #src
