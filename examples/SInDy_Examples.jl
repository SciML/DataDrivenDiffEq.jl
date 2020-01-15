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
tspan = (0.0, 20.0)
prob = ODEProblem(pendulum, u0, tspan)
sol = solve(prob, Tsit5(), saveat = 0.3)

plot(sol)

# Create the differential data
DX = similar(sol[:,:])
for (i, xi) in enumerate(eachcol(sol[:,:]))
    DX[:,i] = pendulum(xi, [], 0.0)
end

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
opt = STRRidge(1e-10/0.05)
Ψ = SInDy(sol[:,:], DX, basis, maxiter = 100, opt = opt)
println(Ψ.basis)


opt = ADMM(1e-10, 0.05)
Ψ = SInDy(sol[:,:], DX, basis, maxiter = 2000, opt = opt)
println(Ψ.basis)


opt = DataDrivenDiffEq.SR3(1e-2, 0.9)

isa( opt, DataDrivenDiffEq.Optimise.AbstractOptimiser)
Ψ = SInDy(sol[:,:], DX, basis, maxiter = 2000, opt = opt)
println(Ψ.basis)
# Transform into ODE System
sys = ODESystem(Ψ)

# Simulate
estimator = ODEProblem(dynamics(Ψ), u0, tspan)
sol_ = solve(estimator, Tsit5(), saveat = sol.t)

# Yeah! We got it right
plot(sol[:,:]')
scatter!(sol_[:,:]')

norm(sol[:,:]-sol_[:,:]) # ≈ 1.89e-13
