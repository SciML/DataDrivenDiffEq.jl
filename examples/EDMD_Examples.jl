using DynamicModeDecomposition
using ModelingToolkit
using DifferentialEquations
using LinearAlgebra
using BenchmarkTools
using Plots
gr()

# Create a basis of functions aka observables ( aka features )
@variables u[1:2]
basis(u) = [u[1]; sin(u[1]); sin(u[2]); u[2]; u[1]*u[2]; u[2]^2]

# Create a test system
function test_discrete(du, u, p, t)
    du[1] = 0.9u[1] + 0.1u[2]^2
    du[2] = sin(u[1]) - 0.1u[1]
end

prob = DiscreteProblem(test_discrete, [1.0; 2.0], (0.0, 10.0))
sol = solve(prob)

plot(sol)

test = ExtendedDMD(sol[:,:], basis)

# Fails here
dudt_ = dynamics(test)

prob_test = DiscreteProblem(dudt_, [1.0; 2.0], (0.0, 10.0))
sol_test = solve(prob_test)

plot(sol)
plot!(sol_test)

plot((sol - sol_test)')

# This still works...
# Get the linear dynamics in koopman space
dψdt = linear_dynamics(test)
ψ_prob = DiscreteProblem(dψdt, test.basis([1.0; 2.0]), (0.0, 10.0))
ψ = solve(ψ_prob)
# Estimate via output
sol_ψ = test.output * ψ
plot(sol_ψ'- sol[:,:]')
