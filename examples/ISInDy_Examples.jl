using DataDrivenDiffEq
using ModelingToolkit
using ProximalOperators
using OrdinaryDiffEq
using LinearAlgebra
using Plots
gr()


# Create a test problem
function simple(u, p, t)
    return [(2.0u[2]^2 - 3.0)/(1.0 + u[1]^2); -u[1]^2/(2.0 + u[2]^2); (1-u[2])/(1+u[3]^2)]
end

#u0 = [2.37; 1.58; -3.10]
u0 =  randn(3)*5.0
tspan = (0.0, 10.0)
prob = ODEProblem(simple, u0, tspan)
sol = solve(prob, Tsit5(), saveat = 0.1)
plot(sol)

# Create the differential data
X = sol[:,:]
DX = similar(X)
for (i, xi) in enumerate(eachcol(X))
    DX[:, i] = simple(xi, [], 0.0)
end

# Create a basis
@variables u[1:3]
polys = ModelingToolkit.Operation[]
# Lots of basis functions
for i ∈ 0:6
    if i == 0
        push!(polys, u[1]^0)
    end
    for ui in u
        if i > 0
            push!(polys, ui^i)
        end
    end
end

basis= Basis(polys, u)

opt = ADM(1e-2)
Ψ = ISInDy(X, DX, basis, opt = opt, maxiter = 10, rtol = 0.1)
println(Ψ.basis)

# Simulate
estimator = ODEProblem(dynamics(Ψ), u0, tspan)
sol_ = solve(estimator, Tsit5(), saveat = 0.1)
plot(sol_)
sol_[:,:] ≈ sol[:,:]
# Yeah! We got it right
plot(sol[:, :]', label = "Original")
plot!(sol_[:,:]', label = "Estimation")
plot(norm.(eachcol(sol[:,:] - sol_[:,:]), 2), label = "Error")

# Full estimation


# Create a test problem
function simple_full(u, p, t)
    du = [-u[1]; -3.0sin(u[2])+u[1]^2]
    return inv(Float64[1 0 ;-1.2 1])*du
end

u0 =  [0.487; -1.53]
tspan = (0.0, 5.0)
prob = ODEProblem(simple_full, u0, tspan)
sol = solve(prob, Tsit5(), saveat = 0.1)
plot(sol)

# Create the differential data
X = sol[:,:]
DX = similar(X)
for (i, xi) in enumerate(eachcol(X))
    DX[:, i] = more_advanced(xi, [], 0.0)
end
Y1 = vcat(X, DX[1, :]')
Y2 = vcat(X, DX[2, :]')

# Create a basis
@variables x[1:2] ẋ[1]
u = vcat(x, ẋ)
polys = ModelingToolkit.Operation[]
# Lots of basis functions
for i ∈ 0:3
    if i == 0
        push!(polys, u[1]^0)
        push!(polys, sin(u[1]))
        push!(polys, sin(u[2]))
        #push!(polys, sin(u[1])*u[1])
        #push!(polys, sin(u[1])*u[2])
    end
    for ui in u
        if i > 0
            push!(polys, ui^i)
        end
    end
end

basis= Basis(polys, u)

# Here the influence of threshold and the sample size can be seen
opt = ADM(1e-2)
Ψ1 = ISInDy(Y1[:, 1:10],basis, opt = opt, nx = 1, maxiter = 50, rtol = 0.99)
println(Ψ1.basis)


opt = ADM(1e-3)
Ψ2 = ISInDy(Y2[:, 1:25],basis, opt = opt, nx = 1, maxiter = 50, rtol = 0.99)
println(Ψ2.basis)
