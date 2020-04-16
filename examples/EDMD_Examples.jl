using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra
using Plots
gr()


# Create a basis of functions aka observables ( aka features )
@variables u[1:2]
h = [u[1]; sin(u[1]); sin(u[2]); u[2]; u[1]*u[2]; u[2]^2]
basis = Basis(h, u)

# Create a test system
function test_discrete(du, u, p, t)
    du[1] = 0.9u[1] + 0.1u[2]^2
    du[2] = sin(u[1]) - 0.1u[1]
end

# Set up the problem
u0 = [1.0; 2.0]
tspan = (0.0, 50.0)
prob = DiscreteProblem(test_discrete, u0, tspan)
sol = solve(prob, FunctionMap())
# Plot the solution
plot(sol)


# Build the edmd with subset of measurements
approximator = EDMD(sol[:,:], basis)

# Lets look at the eigenvalues
scatter(eigvals(approximator))

# Solve the estimation problem
prob_ = DiscreteProblem(approximator, u0, tspan, [])
sol_ = solve(prob_, FunctionMap())

# Plot the error
plot(sol_)
plot(sol.t, abs.(sol - sol_)')
norm(sol-sol_) # ≈ 4.33e-13

# Get the linear dynamics in koopman space
dψ(u, p, t) = operator(approximator)*u
dψ(du, u, p, t) = du .= operator(approximator)*u
ψ0 = basis(u0)
# Simply calling the EDMD struct transforms into the current basis
ψ_prob = DiscreteProblem(dψ, ψ0, tspan)
ψ = solve(ψ_prob, FunctionMap())

# Plot trajectory in edmd basis
plot(sol.t, ψ')
plot(sol.t, basis(Array(sol))')
norm((ψ - basis(Array(sol)))) # ≈ 0.0158

# And in observable space
sol_ψ = outputmap(approximator) * ψ
plot(abs.(sol_ψ'- sol[:,:]'))
norm(sol_ψ - sol) # ≈ 0.00664


# Reduce the basis
reduced_basis = reduce_basis(approximator)
parameters(reduced_basis)
# Reiterate for the reduced approximator with few measurements
reduced_approximator = EDMD(sol[:,1:4], reduced_basis)
operator(reduced_approximator)
# Simply calling the EDMD struct transforms into the current basis
println(reduced_basis)
ψ_prob = DiscreteProblem(reduced_approximator, u0, tspan, [])
ψ = solve(ψ_prob, FunctionMap())

# Plot trajectory in edmd basis with much higher accuracy than before
plot(ψ)
plot!(sol)
#plot!(sol.t, hcat([reduced_approximator(xi) for xi in eachcol(sol)]...)')
norm(ψ - Array(sol)) # ≈ 2.29..


# Update with all other measurements
update!(reduced_approximator, sol[:,8:10], sol[:, 9:11])
ψ_prob = DiscreteProblem(reduced_approximator, u0, tspan, [])
ψ = solve(ψ_prob, FunctionMap())
norm(ψ - Array(sol)) # ≈ 0.133

# Simply calling the EDMD struct transforms into the current basis
ψ_prob = DiscreteProblem(reduced_approximator, u0, tspan, [])
ψ = solve(ψ_prob, FunctionMap())
println(reduced_basis)
# Plot trajectory in edmd basis with much higher accuracy than before
plot(ψ)
plot!(sol)
norm(ψ .- sol) # ≈ 0.134...
