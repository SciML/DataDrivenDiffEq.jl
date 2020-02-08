using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra
using Plots
gr()



# Create a
function pendulum(u, p, t)
    x = u[2]
    y = -9.81sin(u[1]) - 0.1u[2]^3 -0.2*cos(u[1])
    return [x;y]
end

u0 = [0.99π; -1.0]
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
polys = Operation[1]
for i ∈ 1:5
    push!(polys, u.^i...)
    for j ∈ 1:i-1
        push!(polys, u[1]^i*u[2]^j)
    end
end

# And some other stuff
h = [cos(u[1]); sin(u[1]); u[1]*u[2]; u[1]*sin(u[2]); u[2]*cos(u[2]); polys...]

basis = Basis(h, u)
println(basis)

# Get the reduced basis via the sparse regression
# Thresholded Sequential Least Squares, works fine for more data
# than assumptions, converges fast but fails sometimes with too much noise
opt = STRRidge(1e-2)
Ψ = SInDy(sol[:,1:25], DX[:, 1:25], basis, maxiter = 100, opt = opt)
println(Ψ)

# Lasso as ADMM, typically needs more information, more tuning
opt = ADMM(1e-2, 1.0)
Ψ = SInDy(sol[:,1:50], DX[:, 1:50], basis, maxiter = 5000, opt = opt)
println(Ψ)

# SR3, works good with lesser data and tuning
opt = SR3(1e-2, 1.0)
Ψ = SInDy(sol[:,1:30], DX[:, 1:30], basis, maxiter = 5000, opt = opt)
println(Ψ)


# Transform into ODE System
sys = ODESystem(Ψ)

# Simulate
estimator = ODEProblem(dynamics(Ψ), u0, tspan)
sol_ = solve(estimator, Tsit5(), saveat = sol.t)


# Yeah! We got it right
scatter(sol[:,:]')
plot!(sol_[:,:]')
plot(sol.t, abs.(sol-sol_)')
norm(sol[:,:]-sol_[:,:]) # ≈ 1.89e-13
