using DataDrivenDiffEq
using ModelingToolkit
using ProximalOperators
using OrdinaryDiffEq
using LinearAlgebra
using Plots
gr()


# Create a test problem
function simple(u, p, t)
    return [(2.0u[2] - 3.0)/(1.0 + u[1]^2); -u[1]^2/(2.0 + u[2]^2)]
end

function simple_2(u, p, t)
    return -1 .*u ./ (1 .+ u.^2)
end

u0 =  randn(25)*10
tspan = (0.0, 10.0)
prob = ODEProblem(simple_2, u0, tspan)
sol = solve(prob, Tsit5(), saveat = 0.1)

plot(sol)

# Create the differential data
X = sol[:,:]
DX = similar(X)
for (i, xi) in enumerate(eachcol(X))
    DX[:, i] = simple_2(xi, [], 0.0)
end
DX
# Create a basis
@variables u[1:25]
polys = ModelingToolkit.Operation[]
# Lots of basis functions
for i ∈ 0:3
    if i == 0
        push!(polys, u[1]^0)
    end
    for ui in u
        if i > 0
            push!(polys, ui^i)
        end
    end
    #push!(polys, u[2]^i)
    #if i == 1
    #    push!(polys, sin(u[1]))
    #    push!(polys, sin(u[2]))
    #    push!(polys, cos(u[1]))
    #    push!(polys, cos(u[2]))
    #end
end

polys
basis= Basis(polys, u)


svd(X)

opt = ADM(1e-8)
Ψ = ISindy(X, DX, basis, opt = opt, maxiter = 10, rtol = 0.2)
println(Ψ.basis)

# Simulate
estimator = ODEProblem(dynamics(Ψ), u0, tspan)
sol_ = solve(estimator, Tsit5(), saveat = 0.1)
plot(sol_)
# Yeah! We got it right
plot(sol[:, :]', label = "Original")
plot!(sol_[:,:]', label = "Estimation")
plot(norm.(eachcol(sol[:,:] - sol_[:,:]), 2), label = "Error")
savefig("isindy.png")
