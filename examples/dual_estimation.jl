using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra
using Plots
gr()

using Optim

# Based upon the paper
# A unified sparse optimization framework to learn
# parsimonious physics-informed models from data
# Champion et.al.
# https://arxiv.org/pdf/1906.10612.pdf

# Algorithm 3 ( with adaptation to be more general )
# General idea:
# Combine Sparse regression with Optim updates for the parameters

function pendulum(u, p, t)
    x = u[2]
    y = -9.81sin(2.0*u[1]+0.3) #+ 0.5*sin(t*0.1)
    return [x;y]
end

u0 = [0.99π; -1.0]
tspan = (0.0, 10.0)
prob = ODEProblem(pendulum, u0, tspan)
sol = solve(prob, Tsit5(), saveat = 0.1)

plot(sol)

# Create the differential data
DX = similar(sol[:,:])
for (i, xi) in enumerate(eachcol(sol[:,:]))
    DX[:,i] = pendulum(xi, [], 0.0)
end

# Create a basis
@variables u[1:2] t
@parameters p[1:2]

# And some other stuff
h = Operation[cos(p[2]*u[1]+p[1]); u[2]; u[1]]

basis = Basis(h, u, parameters = p, iv = t)

using Optim

# SR3, works good with lesser data and tuning
X = Array(sol)

opt = DataDrivenDiffEq.Optimise.DualOptimiser(basis, LBFGS(), STRRidge(5e-1))

res = SInDy(X, DX, basis, opt = opt, t = sol.t, maxiter = 100)

print(res)
print_equations(res)

sys = ODESystem(res)
dudt = ODEFunction(sys)
ps = parameters(res)

p_ = ODEProblem(dudt, u0, tspan, ps)
sol_ = solve(p_, Tsit5())
plot!(sol_)
