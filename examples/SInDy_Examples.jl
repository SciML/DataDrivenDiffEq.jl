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
sol = solve(prob, Tsit5(),atol = 1e-6, rtol = 1e-6, saveat = 0.1)

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
for i ∈ 1:15
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
λ = 1/norm(DX, Inf)^2 # Good initial guess for threshold
opt = STRRidge(λ)
Ψ = SInDy(sol[:,1:end], DX[:, 1:end], basis, maxiter = 1000, opt = opt)
println(Ψ)

# Parameter tweak for ADMM enables better performance
opt = ADMM(λ, 0.2,1.0)
Ψ = SInDy(sol[:,1:20], DX[:, 1:20], basis, maxiter = 1000, opt = opt)
println(Ψ)

# SR3, works good with more data and tuning
# However, the
# Use SR3 with high relaxation (allows the solution to diverge from LTSQ) and high iterations
opt = SR3(1.0, 10.0)
set_threshold!(opt, λ)
Ψ = SInDy(sol[:,1:25], DX[:, 1:25], basis, maxiter = 5000, opt = opt)
println(Ψ)

# Vary the sparsity threshold with unknown signals
λs = exp10.(-7:0.1:-1)
opt = STRRidge()
Ψ= SInDy(sol[:,1:40], DX[:, 1:40], basis, λs, maxiter = 500, opt = opt)
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
norm(sol[:,:]-sol_[:,:], 2)
@test all(sol[:,:] .≈ sol_[:,:])
