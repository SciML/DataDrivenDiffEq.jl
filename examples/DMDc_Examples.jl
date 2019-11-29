using DataDrivenDiffEq
using DifferentialEquations
using Plots
gr()

# Define measurements from unstable system with known control input
X = [4 2 1 0.5 0.25; 7 0.7 0.07 0.007 0.0007]
U = [-4 -2 -1 -0.5]
B = Float32[1; 0]

# See fail with unknown input
sys = DMDc(X, U)
# But with a little more knowledge
sys = DMDc(X, U, B = B)

# Extract the DMD from inside DMDc
get_dynamics(sys)
# Acess all the other stuff
eigen(sys) .≈ eigen(get_dynamics(sys))
# Get unforced dynamics
dudt_ = dynamics(sys)
prob = DiscreteProblem(dudt_, X[:, 1], (0., 10.))
sol_unforced = solve(prob)
plot(sol_unforced)

# Create a system with cos control input to stabilize
dudt_ = dynamics(sys, control = (u, p, t) -> -0.5u[1])
prob = DiscreteProblem(dudt_, X[:, 1], (0., 10.))
sol = solve(prob)

plot!(sol)
