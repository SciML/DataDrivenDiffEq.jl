# # [Linear Time Continuous System with Controls](@id linear_continuous_controls)
# 
# Now we will extend the [`previous example`](@ref linear_continuous) by adding some exegeneous control signals.
# As always, we will generate some data via `OrdinaryDiffEq.jl`

using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra
using OrdinaryDiffEq
#md using Plots 

A = [-0.9 0.2; 0.0 -0.2]
B = [0.0; 1.0]
u0 = [10.0; -10.0]
tspan = (0.0, 10.0)

f(u, p, t) = A * u .+ B .* sin(0.5 * t)

sys = ODEProblem(f, u0, tspan)
sol = solve(sys, Tsit5(), saveat = 0.05);

# We will use the data provided by our problem, but add the control signal `U = sin(0.5*t)` to it. 
X = Array(sol)
t = sol.t
control(u, p, t) = [sin(0.5 * t)]
prob = ContinuousDataDrivenProblem(X, t, U = control)

# And plot the problems data.

#md plot(prob) 

# Again, we will use `gDMD` to estimate the systems dynamics. Since we have a control signal 
# defined in the problem, the algorithm will detect it automatically and use `gDMDc`:

res = solve(prob, DMDSVD(), digits = 1)
#md println(res) 

# We see that the system has been recovered correctly, indicated by the small error and high AIC score of the result. We can confirm this by looking at the resulting [`Basis`](@ref)

#md 
system = result(res)
#md println(system)

# And also plot the prediction of the recovered dynamics

#md plot(res) 

# Again, we can have a look at the generator of the system, which is independent from the inputs.

generator(system)

# Sticking to the same procedure as earlier, we now use a linear sparse regression to solve the problem

@parameters t
@variables x[1:2](t) u[1:1](t)

basis = Basis([x; u], x, controls = u, independent_variable = t, name = :LinearBasis)
#md print(basis) #hide

# Note that we added a new variable `u[1](t)` as a control to both the equations and the basis constructor. 
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

# Both results can be converted into an `ODESystem`. To include the control signal, we simply 
# substitute the control variables in the corresponding equations.

subs_control = (u[1] => sin(0.5 * t))

eqs = map(equations(sparse_system)) do eq
    eq.lhs ~ substitute(eq.rhs, subs_control)
end

@named sys = ODESystem(eqs,
                       get_iv(sparse_system),
                       states(sparse_system),
                       parameters(sparse_system));

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
@test Array(sol)≈Array(estimate) rtol=5e-2 #src
