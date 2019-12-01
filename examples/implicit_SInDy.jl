using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using DifferentialEquations
using Plots
gr()

function simple(du, u, p, t)
    du[1] = 5 - u[1]^2/(1.0 + u[1])
end

u0 = [5.0]
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
push!(g, u̇[1])
g
basis = Basis(g, [u...; u̇...]);

x = sol[:,:]
dx = similar(x)
Y = []
for (i,xi) in enumerate(eachcol(x))
    dxi = similar(xi)
    simple(dxi, xi, [], 0.0)
    dx[:,i] = dxi
    push!(Y, basis([xi; dxi]))
end

Y = hcat(Y...)


Z = nullspace(Y', atol = Inf)
Q =  ADM(Z, 1e-2, 1e-10, 100000)
Q
scatter(norm.(eachcol(Q), 0), norm.(eachcol(Y'*Q), 2), zcolor = -log.(sqrt.(norm.(eachcol(Q), 0) + norm.(eachcol(Y'*Q), 2))), yaxis = :log)

λ = exp10.(range(-5, stop=-1e-10, length=500))
Q = ADM(Z, λ, 1e-3, 50000)
scatter(norm.(eachcol(Q), 0), norm.(eachcol(Y'*Q), 2), zcolor = -log.(sqrt.(norm.(eachcol(Q), 0) + norm.(eachcol(Y'*Q), 2))), yaxis = :log)
findmin(log.(sqrt.(norm.(eachcol(Q), 0) + norm.(eachcol(Y'*Q), 2))))

Q[:, 3522]
