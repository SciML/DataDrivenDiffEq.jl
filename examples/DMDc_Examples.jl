using DataDrivenDiffEq
using OrdinaryDiffEq
using Plots
gr()

# See the Dynamic Mode Decomposition Book of Kutz et. al.
# Define measurements from unstable system with known control input
X = [4 2 1 0.5 0.25; 7 0.7 0.07 0.007 0.0007]
U = [-4 -2 -1 -0.5]
B = [1.; 0.]

# See fail with unknown input
sys = DMDc(X, U)

# Lets look at the operator
operator(sys)
# True solution is [1.5 0; 0 0.1]

# But with a little more knowledge
sys = DMDc(X, U, B = B)

# Extract the DMD from inside DMDc
dynamics(sys)
# Acess all the other stuff
eigen(sys)
eigvals(sys)
eigvecs(sys)
isstable(sys)
# Get unforced dynamics
dudt_ = dynamics(sys)[2]
prob = DiscreteProblem(dudt_, X[:, 1], (0., 10.))
sol_unforced = solve(prob,  FunctionMap())
plot(sol_unforced)

get_inputmap(sys)(1.0, [], 0.0)
# Create a system with cos control input to stabilize
dudt_ = dynamics(sys, control = (u, p, t) -> -0.5u[1])
prob = DiscreteProblem(dudt_, X[:, 1], (0., 10.))
sol = solve(prob, FunctionMap())

plot!(sol)
