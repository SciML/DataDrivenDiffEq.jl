using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra
using Plots
gr()

function michaelis_menten(u, p, t)
    [0.6 - 1.5u[1]/(0.3+u[1])]
end


u0 = [1.0]
tspan = (0.0, 3.0)
sols = []
problem = ODEProblem(michaelis_menten, u0, tspan)
sol = solve(problem, Tsit5(), saveat = 0.01)


plot(sol)

# Create the differential data
X = sol[:,:]
DX = similar(X)
for (i, xi) in enumerate(eachcol(X))
    DX[:, i] = michaelis_menten(xi, [], 0.0)
end

# Create a basis
@variables u du

basis= Basis(Operation[u; u^2; du; u*du; du*u^2; 1], [u;du])
U = [X; DX]
θ = basis(U)
scatter(sol.t, θ')

using Convex
using SCS
using COSMO

Ξ = Convex.Variable(length(basis), length(basis))
obj = sumsquares(θ' - θ'*Ξ)
smallinds = abs.(rand(length(basis), length(basis))) .== 0
opt = COSMO.Optimizer

for i in 1:10
    if i > 1
        c = [diag(Ξ) == 0; Ξ[smallinds] .== 0]
    else
        c = [diag(Ξ) == 0]
    end
    p = minimize(obj, c)
    Convex.solve!(p, opt, warmstart = i > 1 ? true : false)
    smallinds .= abs.(Ξ.value) .< 3e-1
    println("$i : $(norm(smallinds, 0))")
end


xi = deepcopy(Ξ.value)
xi[abs.(xi) .< 3e-1] .= 0
diag(xi)
heatmap(abs.(xi))

xi[:, 1]

errors = []
for (i,ξ) in enumerate(eachcol(xi))
    # Compute the
    error =  θ[i,:] .- (θ)[i,:]
    push!(errors, error)
end

plot((θ' - θ'*xi))

basis[3]  ~ simplify_constants(sum(xi[:,3].*basis.basis))
