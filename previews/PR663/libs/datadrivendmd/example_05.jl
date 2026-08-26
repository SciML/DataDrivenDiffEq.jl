# # [Using Real Data with Time-Varying Controls](@id real_data_controls)
#
# This example demonstrates how to use DataDrivenDiffEq with real experimental data,
# particularly when you have time-varying control inputs stored in data files.
# This is a common scenario when working with physical systems like RC circuits,
# mechanical systems, or any controlled experiment where inputs vary over time.
#
# ## The Problem Setup
#
# Consider a linear system with controls of the form:
# ```math
# \frac{dx}{dt} = A x + B u
# ```
# where `x` is the state vector, `u` is a time-varying control input, and
# `A` and `B` are constant matrices we want to identify.
#
# In practice, both states `x(t)` and controls `u(t)` are measured at discrete
# time points and stored in data files (e.g., CSV).
#
# ## Simulating Real Data
#
# First, let's create some synthetic "experimental" data that mimics what you might
# load from a CSV file. In a real application, you would load this from your data file
# using packages like CSV.jl and DataFrames.jl.

using DataDrivenDiffEq
using DataDrivenDMD
using LinearAlgebra
using OrdinaryDiffEq
#md using Plots

# Define the true system (unknown in practice)
A_true = [-0.5 0.1; 0.0 -0.3]
B_true = [1.0; 0.5]

# Time-varying control: a combination of sinusoids (simulating real measured control)
function control_signal(t)
    return sin(0.5 * t) + 0.3 * cos(1.2 * t)
end

# System dynamics
function controlled_system!(du, u, p, t)
    ctrl = control_signal(t)
    return du .= A_true * u .+ B_true .* ctrl
end

# Generate "experimental" data
u0 = [1.0, -0.5]
tspan = (0.0, 20.0)
dt = 0.1  # Sampling interval

prob = ODEProblem(controlled_system!, u0, tspan)
sol = solve(prob, Tsit5(), saveat = dt)

# ## Working with Real Data Format
#
# In practice, your data might come from a CSV file with columns like:
# `time, state1, state2, control1`
#
# Here we simulate that data format:

# Time points (like loading from CSV column "time")
t_data = sol.t

# State measurements (like loading from CSV columns "state1", "state2")
# Note: Each column should be a time point, rows are state variables
X_data = Array(sol)

# Control measurements (like loading from CSV column "control1")
# Note: Control values at each time point, must match dimensions
U_data = [control_signal(ti) for ti in t_data]
U_data = reshape(U_data, 1, :)  # Shape: (n_controls, n_timepoints)

# ```julia
# # In practice, you would load data like this:
# using CSV, DataFrames
#
# df = CSV.read("experimental_data.csv", DataFrame)
# t_data = df.time
# X_data = permutedims(Matrix(df[:, [:state1, :state2]]))  # (n_states, n_timepoints)
# U_data = permutedims(Matrix(df[:, [:control1]]))         # (n_controls, n_timepoints)
# ```

# ## Creating the DataDrivenProblem
#
# Now we can create the problem using the data arrays directly.
# The key insight is that `U` can be passed as a matrix of measured values,
# not just as a function!

ddprob = ContinuousDataDrivenProblem(X_data, t_data, U = U_data)

#md plot(ddprob, title = "Data-Driven Problem with Measured Controls")

# ## Solving the Problem
#
# We use DMD with SVD to identify the system dynamics:

res = solve(ddprob, DMDSVD(), digits = 2)

#md println(res) #hide

# ## Examining the Results
#
# The recovered system should approximate our original dynamics:

#md get_basis(res)
#md println(get_basis(res)) #hide

# Let's visualize how well the identified model matches the data:

#md plot(res, title = "Identified System vs Data")

# ## Alternative: Using Control Functions with Interpolation
#
# If you prefer to use a continuous function for controls (e.g., for prediction
# at arbitrary time points), you can interpolate your measured control data.
# The `DataInterpolations.jl` package is useful for this:
#
# ```julia
# using DataInterpolations
#
# # Create an interpolation from your measured control data
# u_interp = LinearInterpolation(vec(U_data), t_data)
#
# # Now you can use it as a control function
# control_func(x, p, t) = [u_interp(t)]
#
# ddprob_interp = ContinuousDataDrivenProblem(X_data, t_data, U = control_func)
# ```
#
# This is particularly useful when:
# - Your control and state measurements are at different time points
# - You want to evaluate the model at times not in your dataset
# - You need smooth derivatives of the control signal

# ## Summary
#
# When working with real experimental data containing time-varying controls:
#
# 1. **Load your data** from CSV or other formats using CSV.jl, DataFrames.jl, etc.
#
# 2. **Format your data** correctly:
#    - States `X`: Matrix of shape `(n_states, n_timepoints)`
#    - Times `t`: Vector of length `n_timepoints`
#    - Controls `U`: Matrix of shape `(n_controls, n_timepoints)`
#
# 3. **Create the problem** using measured control values:
#    ```julia
#    prob = ContinuousDataDrivenProblem(X, t, U = U)
#    ```
#
# 4. **Optionally interpolate** controls if you need a continuous function:
#    ```julia
#    using DataInterpolations
#    u_interp = LinearInterpolation(vec(U), t)
#    control_func(x, p, t) = [u_interp(t)]
#    prob = ContinuousDataDrivenProblem(X, t, U = control_func)
#    ```
#
# 5. **Solve** using your preferred method (DMD, sparse regression, etc.)

#md # ## [Copy-Pasteable Code](@id real_data_controls_copy_paste)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
