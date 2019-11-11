using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using DifferentialEquations
using Plots
gr()

function simple_example(du, u, p, t)
    du[1] = -u[2]/(1.0+0.9*sin(u[1]))
    du[2] = -0.9u[2] + 0.1u[1]
end

# Setup the system
u0 = [0.5; 1.0]
p = [0.6 1.5 0.3]
tspan = (0.0, 4.0)
prob = ODEProblem(simple_example, u0, tspan, p)
sol = solve(prob, saveat = 0.01)
plot(sol)

# Collect the derivatives
x = sol[:,:]
dx = similar(x)
for (i, xi) in enumerate(eachcol(x))
    dxi = similar(xi)
    test(dxi, xi, p, 0.0)
    dx[:,i] = dxi
end

# Define the basis
@parameters t
@variables u[1:2] y[1:2]
# The dictionary
g = [1u[1]; 1u[2]; 1y[1]; 1y[2]; y[1]*y[2]; y[1]*sin(u[1])]
basis = Basis(g, [u...; y...]);

# Approximate the basis
Ξ, scores = ISInDy(x, dx, basis, 1e-10, maxiter = 5)
# Print out the equations
simplify_constants.(Ξ'*basis.basis)
