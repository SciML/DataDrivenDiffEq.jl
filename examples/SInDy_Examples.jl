using DataDrivenDiffEq
using ModelingToolkit
using DifferentialEquations
using Plots
gr()

# Create a test problem
function pendulum(u, p, t)
    x = u[2]
    y = -5sin(u[1]) + 1.0*u[2]^2
    return [x;y]
end

u0 = [0.2π; -1.0]
tspan = (0.0, 10.0)
prob = ODEProblem(pendulum, u0, tspan)
sol = solve(prob)

plot(sol)

# Create the differential data
DX = similar(sol[:,:])
for (i, xi) in enumerate(eachcol(sol[:,:]))
    DX[:,i] = pendulum(xi, [], 0.0)
end

# Create a basis
@variables u[1:2]
polys = [u[1]^0]
for i ∈ 1:3
    for j ∈ 1:3
        push!(polys, u[1]^i*u[2]^j)
    end
end
polys
h = [1u[1];1u[2]; cos(u[1]); sin(u[1]); u[1]*u[2]; u[1]*sin(u[2]); u[2]*cos(u[2]); polys...]
basis = Basis(h, u, [])



#Generate eqs
eqs = SInDy(sol[:,:], DX, basis, ϵ = 1e-1, maxiter = 100)
simplified_expr.(simplify_constants.(eqs))
