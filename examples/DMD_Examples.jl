using DataDrivenDiffEq
using OrdinaryDiffEq
using LinearAlgebra
using Plots
gr()

# Create a linear, time-discrete system
function linear_discrete(du, u, p, t)
    du[1] = 0.9u[1]
    du[2] = 0.9u[2] + 0.1u[1]
end

u0 = [10.0; -2.0]
tspan = (0.0, 10.0)
prob = DiscreteProblem(linear_discrete, u0, tspan)
sol = solve(prob, FunctionMap())

# Create Approximation
approx = DMD(sol[:,:])

# Create the associated problem
prob_approx = DiscreteProblem(approx, u0, tspan)
approx_sol = solve(prob_approx, FunctionMap())

# Show solutions
plot(sol)
plot!(approx_sol)

# Show error
plot((sol .- approx_sol)')
norm(sol .- approx_sol) # ≈ 2.23e-14

# Eigen Decomposition
eigen(approx)
# Eigenvalues
scatter(eigvals(approx), xlim = (-1, 1), ylim = (-1,1))

# Adapt system with new measurements
function linear_discrete_2(du, u, p, t)
    du[1] = 0.9u[1]
    du[2] = 0.9u[2] + 0.2u[1]
end

# Solve the new system
prob2 = DiscreteProblem(linear_discrete_2, u0, (0.0, 20.0))
sol2 = solve(prob2, FunctionMap())

# Split the data
x = sol2[:,1:20]
y = sol2[:,2:21]

# Update our approximation with new measurements
update!(approx, x, y)

# Let's have a look at the operator, which moves near the true value
operator(approx)

# Add time-continuous system
function linear(du, u, p, t)
    du[1] = -0.9*u[1] + 0.1*u[2]
    du[2] = -0.8*u[2]
end

prob_cont = ODEProblem(linear, u0, tspan)
sol_cont = solve(prob_cont, Tsit5())

plot(sol_cont)

X = sol_cont[:,:]
DX = sol_cont(sol_cont.t, Val{1})[:,:]

# Giving the method a time step (which should be sequentially sampled)
# Enables us to get the continuous representation
approx_cont = gDMD(X, DX)
generator(approx_cont)
approx_sys = ODEProblem(approx_cont, u0, tspan)
approx_sol = solve(approx_sys, Tsit5(),  saveat = sol_cont.t)
# Let's have a look at the solution
plot(sol_cont)
plot!(approx_sol)
# And the error
plot(abs.(sol_cont .- approx_sol)')
norm(sol_cont .- approx_sol) # ≈ 1e-13
