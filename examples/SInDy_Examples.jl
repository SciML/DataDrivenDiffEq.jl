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
@parameters w[1:2]
# Lots of polynomials
polys = Operation[1]
for i ∈ 1:5
    push!(polys, u.^i...)
    for j ∈ 1:i-1
        push!(polys, u[1]^i*u[2]^j)
    end
end

# And some other stuff
h = [w[1]*cos(u[1]); w[2]*sin(u[1]); u[1]*u[2]; u[1]*sin(u[2]); u[2]*cos(u[2]); polys...]

basis = Basis(h, u, parameters = w)
println(basis)

# Get the reduced basis via the sparse regression
# Thresholded Sequential Least Squares, works fine for more data
# than assumptions, converges fast but fails sometimes with too much noise
opt = STRRidge(1e-2)
# Enforce all 100 iterations
Ψ = SINDy(sol[:,1:25], DX[:, 1:25], basis, opt, p = [1.0; 1.0], maxiter = 100, convergence_error = 1e-5)
println(Ψ)
print_equations(Ψ)

# Lasso as ADMM, typically needs more information, more tuning
opt = ADMM(1e-2, 1.0)
Ψ = SINDy(sol[:,1:50], DX[:, 1:50], basis, opt, p = [1.0; 1.0], maxiter = 5000, convergence_error = 1e-3, progress = true)
println(Ψ)
print_equations(Ψ)

# Get the associated parameters out of the result
parameters(Ψ)

# SR3, works good with lesser data and tuning
opt = SR3(1e-2, 1.0)
Ψ = SINDy(sol[:,1:end], DX[:, 1:end], basis, opt, p = [0.5; 0.5], maxiter = 5000, convergence_error = 1e-5)
println(Ψ)
print_equations(Ψ, show_parameter = true)


# Vary the sparsity threshold -> gives better results
λs = exp10.(-5:0.1:-1)
# Use SR3 with high relaxation (allows the solution to diverge from LTSQ) and high iterations
opt = SR3(1e-2, 5.0)
Ψ = SINDy(sol[:,1:10], DX[:, 1:10], basis, λs, opt, p = [1.0; 1.0], maxiter = 15000)
println(Ψ)
print_equations(Ψ)

# Transform into ODE System
sys = ODESystem(Ψ)
dudt = ODEFunction(sys)
ps = parameters(Ψ)

# Simulate
estimator = ODEProblem(dudt, u0, tspan, ps)
sol_ = solve(estimator, Tsit5(), saveat = sol.t)

# Yeah! We got it right
scatter(sol.t[1:10], sol[:,1:10]', color = :red, label = nothing)
scatter!(sol.t[11:end], sol[:,11:end]', color = :blue, label = nothing)
plot!(sol_.t, sol_[:, :]', color = :green, label = "Estimation")

plot(sol.t, abs.(sol-sol_)')
norm(sol[:,:]-sol_[:,:], 2)
