# # [Linear Time Continuous System](@id linear_continuous)
# 
# Similar to the [`linear time discrete example`](@ref linear_discrete), we will now estimate a linear time continuous system ``\partial_t u = A u``. 
# We simulate the correspoding system using `OrdinaryDiffEq.jl` and generate a [`ContinuousDataDrivenProblem`](@ref DataDrivenProblem) from the simulated data.

using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenDMD
#md using Plots 
using Test #src

A = [-0.9 0.2; 0.0 -0.2]
u0 = [10.0; -10.0]
tspan = (0.0, 10.0)

f(u, p, t) = A * u

sys = ODEProblem(f, u0, tspan)
sol = solve(sys, Tsit5(), saveat = 0.05);

# We could use the `DESolution` to define our problem, but here we want to use the data for didactic purposes.
# For a [`ContinuousDataDrivenProblem`](@ref DataDrivenProblem), we need either the state trajectory and the timepoints or the state trajectory and its derivate.

X = Array(sol)
t = sol.t
prob = ContinuousDataDrivenProblem(X, t)

# And plot the problems data.

#md plot(prob) 

# We can see that the derivative has been automatically added via a [`collocation`](@ref collocation) method, which defaults to a `LinearInterpolation`. 
# We can do a visual check and compare our derivatives with the interpolation of the `ODESolution`.

#md DX = Array(sol(t, Val{1}))
#md scatter(t, DX', label = ["Solution" nothing], color = :red, legend = :bottomright) 
#md plot!(t, prob.DX', label = ["Linear Interpolation" nothing], color = :black)

# Since we have a linear system, we can use `gDMD`, which approximates the generator of the dynamics

#md res = solve(prob, DMDSVD())
#md println(res) 

# And also plot the prediction of the recovered dynamics

#md plot(res) 

#md # ## [Copy-Pasteable Code](@id linear_continuous_copy_paste)
#md #
#md # ```julia
#md # @__CODE__
#md # ```

@test all(aic(sparse_res) .<= -100.0) #src
@test all(l2error(sparse_res) .<= 5e-1) #src
@test all(determination(sparse_res) .>= 0.97) #src
@test Array(sol)≈Array(estimate) rtol=5e-2 #src
