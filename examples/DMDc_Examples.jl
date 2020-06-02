using DataDrivenDiffEq
using OrdinaryDiffEq
using Plots
gr()

# Define measurements from an unstable system with known control input
X = [4 2 1 0.5 0.25; 7 0.7 0.07 0.007 0.0007]
U = [-4 -2 -1 -0.5]
B = Float32[1; 0]

# See fail with unknown input
sys = DMDc(X, U)
# But with a little more knowledge
sys = DMDc(X, U, B = B)

# Access all the other stuff
eigen(sys)
eigvals(sys)
eigvecs(sys)

prob = DiscreteProblem(sys, X[:, 1], (0., 10.))
sol_unforced = solve(prob,  FunctionMap())
plot(sol_unforced)
sol_unforced[:,:]
