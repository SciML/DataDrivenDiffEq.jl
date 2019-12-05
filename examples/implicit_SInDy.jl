using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using OrdinaryDiffEq
using Plots
gr()

function simple(du, u, p, t)
    du[1] = 0.6 - 1.5*u[1]/(0.3+u[1])
end


u0 = Float64[1.2]
tspan = (0.0, 6.0)
prob = ODEProblem(simple, u0, tspan)
sol = solve(prob, Tsit5())
scatter(sol.t, sol[:,:]')

# Define the basis
@variables u[1] u̇[1]
# The dictionary
g = Array{Operation,1}()

push!(g, [u̇[1] u[1]*u̇[1] u[1]^2*u̇[1] u[1] u[1]^2 u[1]^3 u[1]^4 sin(u[1]) sin(2*u[1]) sin(3*u[1]) ModelingToolkit.Constant(1)]...)

basis = Basis(g, [u...; u̇...]);

x = sol[:,:]
dx = sol(sol.t, Val{1})

λ = exp10.(range(-10, log(3e-1), length= 500))
Z, θ = DataDrivenDiffEq.ISInDy(x, dx[:,:], basis, λ, maxiter = 10000)
sort!(Z, dims = 2, by = x->sqrt(norm(θ'*x, 2)))
Z = Z[:, norm.(eachcol(Z), 0) .> 2 ]
scatter(norm.(eachcol(Z), 0), norm.(eachcol(θ'*Z),2) .+ eps(Float64), yaxis = :log)
q = Z[:,1] / maximum(abs.(Z[:,1]))
Ψ = Basis([simplify_constants(q1'*basis.basis)], basis.variables)
Ψ.basis
