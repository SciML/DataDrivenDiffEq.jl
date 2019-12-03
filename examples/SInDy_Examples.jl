using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra
using Plots
gr()

# Create a test problem
function pendulum(u, p, t)
    x = u[2]
    y = -9.81sin(u[1]) - 0.1u[2]
    return [x;y]
end

u0 = [0.2π; -1.0]
tspan = (0.0, 40.0)
prob = ODEProblem(pendulum, u0, tspan)
sol = solve(prob, Tsit5())

plot(sol)

# Generate the differential
dx = sol(sol.t, Val{1})

# Create a basis
@variables u[1:2]

# Lots of polynomials
polys = [u[1]^0]
for i ∈ 1:3
    for j ∈ 1:3
        push!(polys, u[1]^i*u[2]^j)
    end
end

# And some other stuff
h = [1u[1];1u[2]; cos(u[1]); sin(u[1]); u[1]*u[2]; u[1]*sin(u[2]); u[2]*cos(u[2]); polys...]

basis = Basis(h, u)

# Get the reduced basis via the sparse regression
Ψ = SInDy(sol[:,:], DX, basis, ϵ = 1e-1)

# Transform into ODE System
sys = ODESystem(Ψ)

# Simulate
estimator = ODEProblem(dynamics(Ψ), u0, tspan)
sol_ = solve(estimator, Tsit5(),  saveat = sol.t)

# Compute AIC based on RMSE
AIC(free_parameters(Ψ), sol[:,:], sol_[:,:])
# Compute AIC based on AMSE
AIC(free_parameters(Ψ), sol[:,:], sol_[:,:], likelyhood = (X,Y) -> sum(abs, X - Y))

# Show norm
sum((sol - sol_).^2)
# Yeah! We got it right
plot(sol, vars = (1,2))
plot!(sol, vars = (1,2))

norm(sol-sol_) # ≈ 1.89e-13

# Lorenz system

# Create a test problem
function lorenz!(du, u, p, t)
    du[1] = 10(u[2]-u[1])
    du[2] = u[1]*(28-u[3])-u[2]
    du[3] = u[1]*u[2] - 8/3*u[3]
end

u0 = [0.2; -1.0; 3]
tspan = (0.0, 40.0)
prob = ODEProblem(lorenz!, u0, tspan)
sol = solve(prob, Tsit5())

plot(sol)

dx = sol(sol.t, Val{1})

# Create polynomial basis
@variables x y z

h = Operation[]
for i in 1:3
    push!(h, x^i)
    push!(h, y^i)
    push!(h, z^i)
    for k in 1:2
        push!(h, x^i*y^k)
        push!(h, x^i*z^k)
        push!(h, y^i*z^k)
    end
end

push!(h, ModelingToolkit.Constant(1))

b = Basis(h, [x;y;z])

Ψ = SInDy(sol[:,1:40], dx[:, 1:40], b, ϵ = 1e-1)

# Print the eq system
println(Ψ)

p_ = ODEProblem(dynamics(Ψ), u0,  (0.0, 40.0))
sol_ = solve(p_, Tsit5(), saveat = sol.t)

plot(sol, vars = (1,2,3))
plot!(sol_, vars = (1,2,3))

AIC(free_parameters(Ψ), sol[:,:], sol_[:,:])
