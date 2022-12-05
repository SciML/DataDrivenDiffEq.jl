# # [Linear Time Discrete System](@id linear_discrete)
# 
# We will start by estimating the underlying dynamical system of a time discrete process based on some measurements via [Dynamic Mode Decomposition](https://arxiv.org/abs/1312.0041) on a simple linear system of the form ``u(k+1) = A u(k)``.
# 
# At first, we simulate the correspoding system using `OrdinaryDiffEq.jl` and generate a [`DiscreteDataDrivenProblem`](@ref DataDrivenProblem) from the simulated data.

using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenDMD
#md using Plots

A = [0.9 -0.2; 0.0 0.2]
u0 = [10.0; -10.0]
tspan = (0.0, 11.0)

f(u, p, t) = A * u

sys = DiscreteProblem(f, u0, tspan)
sol = solve(sys, FunctionMap());

# Next we transform our simulated solution into a [`DataDrivenProblem`](@ref). Given that the solution knows its a discrete solution, we can simply write

prob = DataDrivenProblem(sol)

# And plot the solution and the problem 

#md plot(sol, label = string.([:x₁ :x₂])) 
#md scatter!(prob)

# To estimate the underlying operator in the states ``x_1, x_2``, we `solve` the estimation problem using the [`DMDSVD`](@ref) algorithm for approximating the operator. First, we will have a look at the [`DataDrivenSolution`](@ref)

res = solve(prob, DMDSVD(), digits = 1)

# We see that the system has been recovered correctly, indicated by the small error and high AIC score of the result. We can confirm this by looking at the resulting [`Basis`](@ref)

get_basis(res)

# And also plot the prediction of the recovered dynamics

#md plot(res)

#md # ## [Copy-Pasteable Code](@id linear_discrete_copy_paste)
#md #
#md # ```julia
#md # @__CODE__
#md # ```

## Test the result #src
@test rss(res) <= 1e-3 #src
@test r2(res) >= 0.99 #src
@test dof(res) == 3 #src
