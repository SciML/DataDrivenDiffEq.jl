# # [Implicit Nonlinear Dynamics : Michaelis-Menten](@id michaelis_menten)
#
# What if you want to estimate an implicitly defined system of the form ``f(u_t, u, p, t) = 0``?
# The solution : Implicit Sparse Identification. This method was originally described in [this paper](https://ieeexplore.ieee.org/document/7809160/), and currently there exist [robust algorithms](https://royalsocietypublishing.org/doi/10.1098/rspa.2020.0279) to identify these systems.
# We will focus on [Michaelis-Menten Kinetics](https://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics). As before, we will define the [`DataDrivenProblem`](@ref) and the [`Basis`](@ref) containing possible candidate functions for our [sparse regression](@ref sparse_algorithms).
# Let's generate some data! We will use two experiments starting from different initial conditions.

using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenSparse
#md using Plots
using Test #src

function michaelis_menten(u, p, t)
    return [0.6 - 1.5u[1] / (0.3 + u[1])]
end

u0 = [0.5]

ode_problem = ODEProblem(michaelis_menten, u0, (0.0, 4.0));

# Since we have multiple trajectories at hand, we define a [`DataDrivenDataset`](@ref), which collects multiple problems but handles them as a unit
# for the processing.

prob = DataDrivenDataset(
    map(1:2) do i
        solve(
            remake(ode_problem, u0 = i * u0),
            Tsit5(), saveat = 0.1, tspan = (0.0, 4.0)
        )
    end...
)

#md plot(prob)

# Next, we define our [`Basis`](@ref). Since we want to identify an implicit system, we have to include
# some candidate terms which use these as an argument, and inform our constructor about the meaning of these variables.

@parameters t
@variables u(t)[1:1]
u = collect(u)
D = Differential(t)
h = [monomial_basis(u[1:1], 4)...]
basis = Basis([h; h .* (D(u[1]))], u, implicits = D.(u), iv = t)
#md println(basis) #hide

# Next, we define the [`ImplicitOptimizer`](@ref) and `solve` the problem. It wraps a standard optimizer, by default [`STLSQ`](@ref), and performs
# implicit sparse regression upon the selected basis.

opt = ImplicitOptimizer(1.0e-1:1.0e-1:5.0e-1)
res = solve(prob, basis, opt)
#md println(res) #hide

# Let's check the summary statistics of the solution, which show the summary of the residual sum of squares.

summarystats(res)

# We could also check different metrics as described in the [`DataDrivenSolution`](@ref) section, e.g. `aic` or `bic`.
# As we can see, the [`DataDrivenSolution`](@ref) has good metrics.

# Furthermore, inspection of the underlying system shows that the original equations have been recovered correctly:

#md system = get_basis(res)
#md println(system) #hide

#md plot(
#md     plot(prob), plot(res), layout = (1,2)
#md )

#md # ## [Copy-Pasteable Code](@id michaelis_menten_copy_paste)
#md #
#md # ```julia
#md # @__CODE__
#md # ```

## Test #src
