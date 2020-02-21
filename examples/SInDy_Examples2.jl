using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq

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

u0 = [1.0;0.0;0.0]
tspan = (0.0,100.0)
dt = 0.005
prob = ODEProblem(lorenz,u0,tspan)
sol = solve(prob, Tsit5(), saveat = dt)

plot(sol,vars=(1,2,3))

# Differential data from equations
X = Array(sol)
DX = similar(X)
for (i, xi) in enumerate(eachcol(X))
    DX[:,i] = lorenz(xi, [], 0.0)
end

# Estimate differential data from state variables via a Savitzky-Golay filter
# Test on a single variable
windowSize, polyOrder = 9, 4
DX1_cropped, DX1_sg = savitzky_golay(X[1,:], windowSize, polyOrder, deriv=1, dt=dt)

# By default savitzky_golay function crop the borders, where the estimation
# is less accurate. Optionally this can be turn off by passing the
# argument `crop = false` to savitzky_golay function.

# Check if the estimated derivatives are approximate to the "ground truth"
halfWindow = Int(ceil((windowSize+1)/2))
DX = DX[:,halfWindow+1:end-halfWindow]
isapprox(DX1_sg, DX[1,:], rtol=1e-2)

plot(DX1_sg, label = "Estimated with Savitzky-Golay filter")
plot!(DX[1,:],label="Ground truth")


# Now let's estimate the derivatives for all variables and use them infer the equations
X_cropped, DX_sg = savitzky_golay(X, windowSize, polyOrder, deriv=1, dt=dt)

# Create a basis
@variables u[1:3]

# Lots of polynomials
polys = [u[1]^0]
for i ∈ 0:3
    for j ∈ 0:3
        for  k ∈ 0:3
            push!(polys, u[1]^i * u[2]^j * u[3]^k)
        end
    end
end

# And some other stuff
h = [1u[1];1u[2]; cos(u[1]); sin(u[1]); u[1]*u[2]; u[1]*sin(u[2]); u[2]*cos(u[2]); polys...]
basis = Basis(h, u)

# Get the reduced basis via the sparse regression
opt = STRRidge(0.1)
Ψ = SInDy(X_cropped, DX_sg, basis, maxiter = 100, opt = opt)
print(Ψ)


# Let's try adding some noise
using Random
seed = MersenneTwister(3)
X_noisy = X + 0.01*randn(seed,size(X))

X_noisy_cropped, DX_sg = savitzky_golay(X_noisy, windowSize, polyOrder, deriv=1, dt=dt)
X_noisy_cropped, X_smoothed = savitzky_golay(X_noisy, windowSize, polyOrder, deriv=0, dt=dt)

Ψ = SInDy(X_smoothed, DX_sg, basis, maxiter = 100, opt = opt)
print(Ψ)
