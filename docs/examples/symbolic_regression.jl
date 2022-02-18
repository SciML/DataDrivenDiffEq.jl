# # [Symbolic Regression](@id symbolic_regression_simple)
# 
# DataDrivenDiffEq offers an interface to [`SymbolicRegression.jl`](https://github.com/MilesCranmer/SymbolicRegression.jl) to infer more complex functions. To 
# use it, simply load a sufficient version of `SymbolicRegression` (currently we supported version 0.6.14 to 0.6.19).

using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra
using OrdinaryDiffEq
using SymbolicRegression
#md using Plots 

A = [-0.9 0.2; 0.0 -0.2]
B = [0.0; 1.0]
u0 = [10.0; -10.0]
tspan = (0.0, 10.0)

f(u,p,t) = A*u .+ B .* sin(0.5*t)

sys = ODEProblem(f, u0, tspan)
sol = solve(sys, Tsit5(), saveat = 0.05);

# We will use the data provided by our problem, but add the control signal `U = sin(0.5*t)` to it. Instead of using a function, like in [another example](@ref linear_continuous_control)
X = Array(sol) 
t = sol.t 
U = permutedims(sin.(0.5*t))
prob = ContinuousDataDrivenProblem(X, t, U = U)

# And plot the problems data.

#md plot(prob) 

# To solve our problem, we will use [`EQSearch`](@ref), which provides a wrapper for the symbolic regression interface.
# By default, it takes in a `Vector` of `Functions` and additional [keyworded arguments](https://astroautomata.com/SymbolicRegression.jl/v0.6/api/#Options). For now, we will stick to simple operations 
# like addition, subtraction and multiplication, use a `L1DistLoss` and limit the maximum depth of the equation trees.

alg = EQSearch([+, *, -], loss = L1DistLoss(), maxdepth = 2)

# Again, we `solve` the problem to obtain a [`DataDrivenResult`](@ref). Note that any additional keyworded arguments are passed onto 
# symbolic regressions [`EquationSearch`](https://astroautomata.com/SymbolicRegression.jl/v0.6/api/#EquationSearch)

res = solve(prob, alg, numprocs = 0, multithreading = false)
#md println(res) 

# We see that the system has been recovered correctly, indicated by the small error. A closer look at the equations r

system = result(res)
#md println(system)

# Shows that while not obvious, the representation 
# And also plot the prediction of the recovered dynamics

#md plot(res) 

# To convert the result into an `ODESystem`, we substitute the control signal

u = controls(system)
t = get_iv(system)

subs_control = (u[1] => sin(0.5*t))

eqs = map(equations(system)) do eq
    eq.lhs ~ substitute(eq.rhs, subs_control)
end

@named sys = ODESystem(
    eqs, 
    get_iv(system),
    states(system),
    []
    );

# And simulated using `OrdinaryDiffEq.jl` using the (known) initial conditions and the parameter mapping of the estimation.
# Since the parameters are *hard numerical values* we do not need to include those.

x = states(system)
x0 = [x[1] => u0[1], x[2] => u0[2]]

ode_prob = ODEProblem(sys, x0, tspan)
estimate = solve(ode_prob, Tsit5(), saveat = prob.t);

# And look at the result
#md plot(sol, color = :black)
#md plot!(estimate, color = :red, linestyle = :dash)

#md # ## [Copy-Pasteable Code](@id symbolic_regression_simple_copy_paste)
#md #
#md # ```julia
#md # @__CODE__
#md # ```

@test all(l2error(res) .<= 5e-1) #src
@test Array(sol) ≈ Array(estimate) rtol = 5e-2 #src
