using DataDrivenDiffEq
using ModelingToolkit
using DifferentialEquations
using Plots
gr()

# Create a basis of functions aka observables ( aka features )
@variables u[1:2]
basis = [u[1]; sin(u[1]); sin(u[2]); u[2]; u[1]*u[2]; u[2]^2]

# Create a test system
function test_discrete(du, u, p, t)
    du[1] = 0.9u[1] + 0.1u[2]^2
    du[2] = sin(u[1]) - 0.1u[1]
end

# Set up the problem
u0 = [1.0; 2.0]
tspan = (0.0, 10.0)
prob = DiscreteProblem(test_discrete, [1.0; 2.0], (0.0, 10.0))
sol = solve(prob)
# Plot the solution
plot(sol)

reduce_basis!(approximator)

# Build the edmd
approximator = ExtendedDMD(sol[:,:], basis)
reduce(approximator)


scatter(eigvals(approximator))

# Get the nonlinear dynamics
dudt_ = dynamics(approximator)

prob_test = DiscreteProblem(dudt_, [1.0; 2.0], (0.0, 10.0))
sol_test = solve(prob_test)

plot(sol)
plot!(sol_test)

plot(sol.t, (sol - sol_test)')

# Get the linear dynamics in koopman space
dψdt = linear_dynamics(approximator)
ψ_prob = DiscreteProblem(dψdt, approximator([1.0; 2.0]), (0.0, 10.0))
ψ = solve(ψ_prob, saveat = sol.t)

# Plot trajectory in basis
plot(sol.t, ψ')
plot(sol.t, hcat([approximator(xi) for xi in eachcol(sol)]...)')

# And in observable space
sol_ψ = test.output * ψ
plot(sol_ψ'- sol[:,:]')
