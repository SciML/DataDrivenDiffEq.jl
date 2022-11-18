# # [Linear Time Continuous System with Controls](@id linear_continuous_controls)
# 
# Now we will extend the [`previous example`](@ref linear_continuous) by adding some exegeneous control signals.
# As always, we will generate some data via `OrdinaryDiffEq.jl`

using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenDMD
#md using Plots 

A = [-0.9 0.2; 0.0 -0.2]
B = [0.0; 1.0]
u0 = [10.0; -10.0]
tspan = (0.0, 10.0)

f(u,p,t) = A*u .+ B .* sin(0.5*t)

sys = ODEProblem(f, u0, tspan)
sol = solve(sys, Tsit5(), saveat = 0.05);

# We will use the data provided by our problem, but add the control signal `U = sin(0.5*t)` to it. 
X = Array(sol) 
t = sol.t 
control(u,p,t) = [sin(0.5*t)]
prob = ContinuousDataDrivenProblem(X, t, U = control)

# And plot the problems data.

#md plot(prob) 

# Again, we will use `gDMD` to estimate the systems dynamics. Since we have a control signal 
# defined in the problem, the algorithm will detect it automatically and use `gDMDc`:

res = solve(prob, DMDSVD(), digits = 1)

# We see that the system has been recovered correctly, indicated by the small error and high AIC score of the result. We can confirm this by looking at the resulting [`Basis`](@ref)
# And also plot the prediction of the recovered dynamics

#md plot(res) 

#md # ## [Copy-Pasteable Code](@id linear_continuous_copy_paste)
#md #
#md # ```julia
#md # @__CODE__
#md # ```

#src #@test all(aic(sparse_res) .<= -200.0) #src
#src #@test all(l2error(sparse_res) .<= 5e-1) #src
#src #@test all(determination(sparse_res) .>= 0.97) #src
#src #@test Array(sol) ≈ Array(estimate) rtol = 5e-2 #src