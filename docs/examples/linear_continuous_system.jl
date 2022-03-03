# # [Linear Time Continuous System](@id linear_continuous)
# 
# Similar to the [`linear time discrete example`](@ref linear_discrete), we will now estimate a linear time continuous system ``\partial_t u = A u``. 
# We simulate the correspoding system using `OrdinaryDiffEq.jl` and generate a [`ContinuousDataDrivenProblem`](@ref DataDrivenProblem) from the simulated data.

using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra
using OrdinaryDiffEq
#md using Plots 

A = [-0.9 0.2; 0.0 -0.2]
u0 = [10.0; -10.0]
tspan = (0.0, 10.0)

f(u,p,t) = A*u

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

# We see that the system has been recovered correctly, indicated by the small error and high AIC score of the result. We can confirm this by looking at the resulting [`Basis`](@ref)

#md system = result(res)
#md println(system)

# And also plot the prediction of the recovered dynamics

#md plot(res) 

# Or a have a look at the metrics of the result

#md metrics(res) 

# And check the parameters of the result 

#md parameters(res)

# or the generator of the system

#md Matrix(generator(system))

# to see that the operator is slightly off, but within expectations. 
# In a real example, this could have many reasons, e.g. noisy data, insufficient time samples or missing states.

# Sticking to the same procedure as earlier, we now use a linear sparse regression to solve the problem

using ModelingToolkit

@parameters t
@variables x[1:2](t)

basis = Basis(x, x, independent_variable = t, name = :LinearBasis)
#md print(basis) #hide


# Afterwards, we simply `solve` the already defined problem with our `Basis` and a `SparseOptimizer`

sparse_res = solve(prob, basis, STLSQ(1e-1))
#md println(sparse_res)

# Which holds the same equations
sparse_system = result(sparse_res)
#md println(sparse_system)

# Again, we can have a look at the result

#md plot(
#md     plot(prob), plot(sparse_res), layout = (1,2)
#md )

# Both results can be converted into an `ODESystem`

@named sys = ODESystem(
    equations(sparse_system), 
    get_iv(sparse_system),
    states(sparse_system), 
    parameters(sparse_system)
    );

# And simulated using `OrdinaryDiffEq.jl` using the (known) initial conditions and the parameter mapping of the estimation.

x0 = [x[1] => u0[1], x[2] => u0[2]]
ps = parameter_map(sparse_res)

ode_prob = ODEProblem(sys, x0, tspan, ps)
estimate = solve(ode_prob, Tsit5(), saveat = prob.t);

# And look at the result
#md plot(sol, color = :black)
#md plot!(estimate, color = :red, linestyle = :dash)

#md # ## [Copy-Pasteable Code](@id linear_continuous_copy_paste)
#md #
#md # ```julia
#md # @__CODE__
#md # ```

@test all(aic(sparse_res) .>= 1e3) #src
@test all(l2error(sparse_res) .<= 5e-1) #src
@test all(determination(sparse_res) .>= 0.97) #src
@test Array(sol) ≈ Array(estimate) rtol = 5e-2 #src
