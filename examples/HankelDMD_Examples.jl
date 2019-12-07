using DataDrivenDiffEq
using OrdinaryDiffEq
using LinearAlgebra
using Plots
gr()

# Create a linear , time discrete system
function linear_discrete(du, u, p, t)
    du[1] = 0.9u[1]
    du[2] = 0.9u[2] + 0.1*u[1]
end

u0 = [10.0; -2.0]
tspan = (0.0, 30.0)
prob = DiscreteProblem(linear_discrete, u0, tspan)
sol = solve(prob, FunctionMap())
plot(sol)

# Create Approximation
approx = HankelDMD(sol[:,:], 5, 4)
operator(approx)
# Create a test function
approx_dudt = dynamics(approx)
# Create the associated problem
prob_approx = DiscreteProblem(approx_dudt, u0, tspan)
approx_sol = solve(prob_approx, FunctionMap())

# Show solutions
plot(sol)
plot!(approx_sol)
# Show error
plot((sol .- approx_sol)')

# Eigen Decomposition
eigen(approx)
# Stability?
isstable(approx)
# Eigenvalues
scatter(eigvals(approx), xlim = (-1, 1), ylim = (-1,1))

# Adapt system with new measurements
function linear_discrete_2(du, u, p, t)
    du[1] = 0.9u[1]
    du[2] = 0.9u[2] + 0.2u[1]
end

prob2 = DiscreteProblem(linear_discrete_2, u0, (0.0, 20.0))
sol2 = solve(prob2)

x = sol2[:,1:20]
y = sol2[:,2:21]

# TODO needs scaling based on error

update!(approx, x, y)
# Moves near true value
approx.Ã

# Add time continouos system
function linear(du, u, p, t)
    du[1] = -0.9*u[1] + 0.1*u[2]
    du[2] = -0.8*u[2]
end

prob_cont = ODEProblem(linear, u0, tspan)
sol_cont = solve(prob_cont, saveat = 0.1)

plot(sol_cont)

approx_cont = ExactDMD(sol_cont[:,:], Δt = 0.1)

test = dynamics(approx_cont, discrete = false)

approx_sys = ODEProblem(test, u0, tspan)
approx_sol = solve(approx_sys, saveat = 0.1)

plot(sol_cont)
plot!(approx_sol)
plot(abs.(sol_cont .- approx_sol)')
