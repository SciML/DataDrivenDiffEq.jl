using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using DifferentialEquations
using Plots
gr()

function simple(du, u, p, t)
    du[1] = -sin(u[1])+3.0
end

u0 = Float64[π/2]
tspan = (0.0, 10.0)
prob = ODEProblem(simple, u0, tspan)
sol = solve(prob, Tsit5(), saveat = 0.1)
plot(sol)

# Define the basis
@parameters t
@variables u[1] u̇[1]
# The dictionary
g = Array{Operation,1}()
push!(g, ModelingToolkit.Constant(1))
push!(g, [u[1] u̇[1] u[1]*u̇[1] u[1]^2 u[1]^3 u[1]^4 sin(u[1])]...)

basis = Basis(g, [u...; u̇...]);
basis


x = sol[:,:]
dx = similar(x)
Y = []
for (i,xi) in enumerate(eachcol(x))
    dxi = similar(xi)
    simple(dxi, xi, [], 0.0)
    dx[:,i] = dxi
    push!(Y, basis([xi; dxi]))
end

Z = DataDrivenDiffEq.ISInDy(x, dx, basis, 5e-3)
Z.basis
