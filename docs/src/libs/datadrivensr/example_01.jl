# # [Symbolic Regression of a Linear Time Continuous Systems](@id symbolic_regression_simple)
# 
# !!! note 
#
#   Symbolic regression is using regularized evolution, simulated annealing, and gradient-free optimization to find suitable equations. 
#   Hence, the performance might differ and depends strongly on the hyperparameters of the optimization. 
#   This example might not recover the groundtruth, but is showing off the use within `DataDrivenDiffEq.jl`.
#
# DataDrivenDiffEq offers an interface to [`SymbolicRegression.jl`](https://docs.sciml.ai/SymbolicRegression/stable/) to infer more complex functions.

using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenSR
#md using Plots 

A = [-0.9 0.2; 0.0 -0.5]
B = [0.0; 1.0]
u0 = [10.0; -10.0]
tspan = (0.0, 20.0)

f(u, p, t) = A * u .+ B .* sin(0.5 * t)

sys = ODEProblem(f, u0, tspan)
sol = solve(sys, Tsit5(), saveat = 0.01);

# We will use the data provided by our problem, but add the control signal `U = sin(0.5*t)` to it. Instead of using a function, like in [another example](@ref linear_continuous_controls)
X = Array(sol)
t = sol.t
U = permutedims(sin.(0.5 * t))
prob = ContinuousDataDrivenProblem(X, t, U = U)

# And plot the problems data.

#md plot(prob) 

# To solve our problem, we will use [`EQSearch`](@ref), which provides a wrapper for the [symbolic regression interface](https://astroautomata.com/SymbolicRegression.jl/v0.6/api/#Options). 
# We will stick to simple operations, use a `L1DistLoss`, and limit the verbosity of the algorithm. 

eqsearch_options = SymbolicRegression.Options(binary_operators = [+, *],
                                              loss = L1DistLoss(),
                                              verbosity = -1, progress = false, npop = 30)

alg = EQSearch(eq_options = eqsearch_options)

# Again, we `solve` the problem to obtain a [`DataDrivenSolution`](@ref). Note that any additional keyworded arguments are passed onto 
# symbolic regressions [`EquationSearch`](https://astroautomata.com/SymbolicRegression.jl/v0.6/api/#EquationSearch) with the exception of `niterations` which 
# is `maxiters`

res = solve(prob, alg, options = DataDrivenCommonOptions(maxiters = 100))
#md println(res) 

# We can inspect the systems metrics, here the `loglikelihood` of the result.

loglikelihood(res)

# !!! note 
#
#   Currently the parameters of the result found by [`EQSearch`](@ref) are not turned into symbolic parameters.
#   This affects some functions like `dof`, `aicc`, `bic`. 

system = get_basis(res)
#md println(system) # hide

#md # ## [Copy-Pasteable Code](@id symbolic_regression_simple_copy_paste)
#md #
#md # ```julia
#md # @__CODE__
#md # ```

@test rss(res) .<= 5e-1 #src
@test r2(res) >= 0.95 #src
