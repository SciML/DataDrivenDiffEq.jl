using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra
using Plots
gr()


# Create a test problem
function simple(u, p, t)
    return [(2.0u[2]^2 - 3.0)/(1.0 + u[1]^2); -u[1]^2/(2.0 + u[2]^2); (1-u[2])/(1+u[3]^2)]
end

u0 = [2.37; 1.58; -3.10]
tspan = (0.0, 10.0)
prob = ODEProblem(simple, u0, tspan)
sol = solve(prob, Tsit5(), saveat = 0.1)
plot(sol)

# Create the differential data
X = sol[:,:]
DX = similar(X)
for (i, xi) in enumerate(eachcol(X))
    DX[:, i] = simple(xi, [], 0.0)
end

# Create a basis
@variables u[1:3]
polys = [monomial_basis(u, 6);1]

basis= Basis(polys, u)
Ψ = ISINDy(X, DX, basis, ADM(1e-2), maxiter = 10, rtol = 0.1)
println(Ψ)
print_equations(Ψ)

# Lets use SINDy pi for a parallel implicit implementation
# Create a basis 
@variables u[1:3]
polys = [monomial_basis(u, 10); 1]

basis= Basis(polys, u)

# Simply use any optimizer you would use for SINDy
Ψ = ISINDy(X[:, :], DX[:, :], basis, STRRidge(1e-1), maxiter = 100, normalize = true)
println(Ψ)
print_equations(Ψ)

# Transform into ODE System
sys = ODESystem(Ψ)
dudt = ODEFunction(sys)
ps = parameters(Ψ)

# Simulate
estimator = ODEProblem(dudt, u0, tspan, ps)
sol_ = solve(estimator, Tsit5(), saveat = 0.1)

# Yeah! We got it right
plot(sol.t[:], sol[:,:]', color = :red, label = nothing)
plot!(sol_.t, sol_[:, :]', color = :green, label = "Estimation")

plot(sol.t, abs.(sol-sol_)')
norm(sol[:,:]-sol_[:,:], 2) # approx 1.2e-13
