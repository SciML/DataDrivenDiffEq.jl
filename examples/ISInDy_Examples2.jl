# Based upon the paper for parallel implicit sindy
using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra
using Plots
gr()


function cart_pole(u, p, t)
    du = similar(u)
    F = -0.2 + 0.5*sin(6*t) #no input for now
    du[1] = u[3]
    du[2] = u[4]
    du[3] = -(19.62*sin(u[1])+sin(u[1])*cos(u[1])*u[3]^2+F*cos(u[1]))/(2-cos(u[1])^2)
    du[4] = -(sin(u[1])*u[3]^2 + 9.81*sin(u[1])*cos(u[1])+F)/(2-cos(u[1])^2)
    return du
end

u0 = [0.3; 0; 1.0; 0]
tspan = (0.0, 16.0)
dt = 0.001
cart_pole_prob = ODEProblem(cart_pole, u0, tspan)
solution = solve(cart_pole_prob, Tsit5(), saveat = dt)

plot(solution)

# Create the differential data
X = solution[:,:]
DX = similar(X)
for (i, xi) in enumerate(eachcol(X))
    DX[:, i] = cart_pole(xi, [], solution.t[i])
end

plot(DX')

@variables u[1:4] t
polys = Operation[]
# Lots of basis functions -> sindy pi can handle more than ADM()
for i ∈ 0:4
    if i == 0
        push!(polys, u[1]^0)
    else
        if i < 2
            push!(polys, u.^i...)
        else
            push!(polys, u[3:4].^i...)
        end
        
    end
end
push!(polys, sin.(u[1])...)
push!(polys, cos.(u[1]))
push!(polys, sin.(u[1]).*u[3:4]...)
push!(polys, sin.(u[1]).*u[3:4].^2...)
push!(polys, cos.(u[1]).^2...)
push!(polys, sin.(u[1]).*cos.(u[1])...)
push!(polys, sin.(u[1]).*cos.(u[1]).*u[3:4]...)
push!(polys, sin.(u[1]).*cos.(u[1]).*u[3:4].^2...)
push!(polys, -0.2+0.5*sin(6*t))
basis= Basis(polys, u, iv = t)

# Simply use any optimizer you would use for sindy
λ = exp10.(-4:0.1:-1)
g(x) = norm([1e-3; 10.0] .* x, 2)
Ψ = ISInDy(X[:,:], DX[:, :], basis, λ, STRRidge(), maxiter = 100, normalize = false, t = solution.t, g = g)
println(Ψ)
print_equations(Ψ, show_parameter = true)

# Transform into ODE System
sys = ODESystem(Ψ)
dudt = ODEFunction(sys)
ps = parameters(Ψ)

# Simulate
estimator = ODEProblem(dudt, u0, tspan, ps)
sol_ = solve(estimator, Tsit5(), saveat = dt)

# Yeah! We got it right
plot(solution.t[:], solution[:,:]', color = :red, label = nothing)
plot!(sol_.t, sol_[:, :]', color = :green, label = "Estimation")

plot(solution.t, abs.(solution-sol_)')
norm(solution[:,:]-sol_[:,:], 2) # ≈ 0.018
