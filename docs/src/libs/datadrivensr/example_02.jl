# # [Symbolic Regression of a Nonlinear System via Lifting](@id symbolic_regression_lifted)
#
# To infer more complex examples, [`EQSearch`](@ref) also can be called with a [`Basis`](@ref) to use
# predefined features. Let's look at the well-known pendulum model

using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenSR
#md using Plots

function pendulum!(du, u, p, t)
    du[1] = u[2]
    du[2] = -9.81 * sin(u[1])
end

u0 = [0.1, π / 2]
tspan = (0.0, 10.0)
sys = ODEProblem{true, SciMLBase.NoSpecialize}(pendulum!, u0, tspan)
sol = solve(sys, Tsit5());

# We will use the data provided by our problem, but add the control signal `U = sin(0.5*t)` to it. Instead of using a function, like in [another example](@ref linear_continuous_controls)
prob = DataDrivenProblem(sol)

# And plot the problems data.

#md plot(prob)

# To solve our problem, we will use [`EQSearch`](@ref), which provides a wrapper for the [symbolic regression interface](https://astroautomata.com/SymbolicRegression.jl/v0.6/api/#Options).
# We will stick to simple operations, use a `L1DistLoss`, and limit the verbosity of the algorithm.
# Note that we do not include `sin`, but rather lift the search space of variables.

@variables u[1:2]
u = collect(u)

basis = Basis([polynomial_basis(u, 2); sin.(u)], u)

eqsearch_options = SymbolicRegression.Options(binary_operators = [+, *],
    loss = L1DistLoss(),
    verbosity = -1, progress = false, npop = 30,
    timeout_in_seconds = 60.0)

alg = EQSearch(eq_options = eqsearch_options)

# Again, we `solve` the problem to obtain a [`DataDrivenSolution`](@ref) with similar options as the [previous example](@ref symbolic_regression_simple) but
# provide a [`Basis`](@ref) along the arguments.

res = solve(prob, basis, alg, options = DataDrivenCommonOptions(maxiters = 100))
#md println(res)

# !!! note
#
#    Currently, the parameters of the result found by [`EQSearch`](@ref) are not turned into symbolic parameters.
#    This affects some functions like `dof`, `aicc`, `bic`.

system = get_basis(res)
#md println(system) # hide

#md # ## [Copy-Pasteable Code](@id symbolic_regression_simple_copy_paste)
#md #
#md # ```julia
#md # @__CODE__
#md # ```

## Test #src
@test rss(res) .<= 5e-2 #src
@test r2(res) >= 0.95 #src
