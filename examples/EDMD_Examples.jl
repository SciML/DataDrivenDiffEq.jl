using DynamicModeDecomposition
using DifferentialEquations
using LinearAlgebra
using BenchmarkTools
using Plots
gr()

# Create a basis of functions aka observables ( aka features )
b₀ = BasisFunction("x[1]")
b₁ = BasisFunction("sin(x[1])")
b₂ = BasisFunction("x[2]")
b₄ = BasisFunction("x[1]*x[2]")
b₅ = BasisFunction("x[2]^2")

# Create a Candidate basis
c = BasisCandidate([b₀ b₂ b₁])
push!(c, b₄)
push!(c, b₅)

# Create a test system
function test_discrete(du, u, p, t)
    du[1] = 0.9u[1] + 0.1u[2]^2
    du[2] = sin(u[1]) - 0.1u[1]
end

prob = DiscreteProblem(test_discrete, [1.0; 2.0], (0.0, 10.0))
sol = solve(prob)

plot(sol)

test = ExtendedDMD(sol[:,:], c)
dudt_ = dynamics(test)

prob_test = DiscreteProblem(dudt_, [1.0; 2.0], (0.0, 10.0))
sol_test = solve(prob_test)

plot(sol)
plot!(sol_test)

plot((sol - sol_test)')


# Get the linear dynamics in koopman space
dψdt = linear_dynamics(test)
ψ_prob = DiscreteProblem(dψdt, test.basis([1.0; 2.0]), (0.0, 10.0))
ψ = solve(ψ_prob)
# Estimate via output
sol_ψ = test.output * ψ
plot(sol_ψ'- sol[:,:]')
