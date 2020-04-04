using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq

using Dierckx
using Statistics
using LinearAlgebra
using Plots
gr()

# Create a test problem
function lorenz(u,p,t)
    x, y, z = u

    ẋ = 10.0*(y - x)
    ẏ = x*(28.0-z) - y
    ż = x*y - (8/3)*z
    return [ẋ, ẏ, ż]
end

u0 = [-8.;8.;27.]
tspan = (0.0,200.0)
dt = 0.001
prob = ODEProblem(lorenz,u0,tspan)
sol = solve(prob, Tsit5(), reltol = 1e-12, abstol = 1e-12,  saveat = dt)

plot(sol,vars=(1,2,3))

# Differential data from equations
X = Array(sol)


# Time delay coordinates
stackmax = 100

H = zeros(eltype(X), stackmax, size(X,2)-stackmax)
for i in 1:stackmax
    H[i, :] = X[1, i:end-stackmax+i-1]
end


m,n = minimum(size(H)), maximum(size(H))
U, S, V = svd(H, full = false)
τ = DataDrivenDiffEq.optimal_svht(m,n)
r = length(S[S .> τ*median(S)])
r = minimum([15, r])

z = Array(V[:, 1:r]')
dz = similar(z)

for (i, vi) in enumerate(eachrow(z))
    x_int = Spline1D(sol.t[1:length(vi)], vi)
    dz[i, :] = derivative(x_int, sol.t[1:length(vi)])
end


plot(plot(z[1, :]),plot(z[end, :].^2),  layout = (2, 1))

@variables u[1:r]

basis = Basis(u, u)
opt = SR3(1e-1)
b = SInDy(z[:, 1:end], dz[1:end-1, 1:end], basis, maxiter = 1000, opt = opt, normalize = true)

# Coincides with the paper results
println(b)
