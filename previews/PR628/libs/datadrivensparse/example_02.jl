# # [Sparse Identification with noisy data](@id noisy_sindy)
#
# Many real-world data sources are corrupted with measurement noise, which can have
# a big impact on the recovery of the underlying equations of motion. This example shows how we can
# use a [collocation method](@ref collocation) and [batching](@ref dataprocessing) to perform SINDy in the presence of
# noise.

using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DataDrivenSparse
using StableRNGs
#md using Plots
#md gr()

rng = StableRNG(1337)

function pendulum(u, p, t)
    x = u[2]
    y = -9.81sin(u[1]) - 0.3u[2]^3 - 3.0 * cos(u[1]) - 10.0 * exp(-((t - 5.0) / 5.0)^2)
    return [x; y]
end

u0 = [0.99π; -1.0]
tspan = (0.0, 15.0)
prob = ODEProblem(pendulum, u0, tspan)
sol = solve(prob, Tsit5(), saveat = 0.01);

# We add random noise to our measurements.

X = sol[:, :] + 0.2 .* randn(rng, size(sol));
ts = sol.t;

#md plot(ts, X', color = :red)
#md plot!(sol, color = :black)

# To estimate the system, we first create a [`DataDrivenProblem`](@ref), which requires measurement data.
# Using a [collocation method](@ref collocation), it automatically provides the derivative and smoothes the trajectory. Control signals can be passed
# in as a function `(u,p,t)->control` or an array of measurements.

prob = ContinuousDataDrivenProblem(
    X, ts, GaussianKernel(),
    U = (u, p, t) -> [exp(-((t - 5.0) / 5.0)^2)],
    p = ones(2)
)

#md plot(prob, size = (600,600))

# Now we infer the system structure. First we define a [`Basis`](@ref) which collects all possible candidate terms.
# Since we want to use SINDy, we call `solve` with an [`sparsifying algorithm`](@ref sparse_algorithms), in this case [`STLSQ`](@ref) which iterates different sparsity thresholds
# and returns a Pareto optimal solution. Note that we include the control signal in the basis as an additional variable `c`.

@variables u[1:2] c[1:1]
@parameters w[1:2]
u = collect(u)
c = collect(c)
w = collect(w)

h = Num[sin.(w[1] .* u[1]); cos.(w[2] .* u[1]); polynomial_basis(u, 5); c]

basis = Basis(h, u, parameters = w, controls = c);
println(basis) # hide

# To solve the problem, we also define a [`DataProcessing`](@ref) which defines randomly shuffled minibatches of our data and selects the
# best fit.

sampler = DataProcessing(split = 0.8, shuffle = true, batchsize = 30, rng = rng)
λs = exp10.(-10:0.1:0)
opt = STLSQ(λs)
res = solve(
    prob, basis, opt,
    options = DataDrivenCommonOptions(data_processing = sampler, digits = 1)
)
#src println(res) #hide

# !!! info
#     A more detailed description of the result can be printed via `print(res, Val{true})`, which also includes the discovered equations and parameter values.
#
# Where the resulting [`DataDrivenSolution`](@ref) stores information about the inferred model and the parameters:

system = get_basis(res)
params = get_parameter_map(system)
println(system) # hide
println(params) # hide

# We can see that even if there are other terms active, the most important terms are included inside the model.

# And a visual check of the result can be performed by plotting the result

#md plot(
#md     plot(prob), plot(res), layout = (1,2)
#md )

#md # ## [Copy-Pasteable Code](@id autoregulation_copy_paste)
#md #
#md # ```julia
#md # @__CODE__
#md # ```

## Test #src
