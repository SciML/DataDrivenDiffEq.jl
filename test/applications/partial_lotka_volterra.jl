using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra
@info "Loading DiffEqSensitivity"
using DiffEqSensitivity
@info "Loading Optim"
using Optim
@info "Loading DiffEqFlux"
using DiffEqFlux
@info "Loading Flux"
using Flux
@info "Loading OrdinaryDiffEq"
using OrdinaryDiffEq
using Test
@info "Finished loading packages"


const losses = []


@info "Started Lotka Volterra UODE Testset"
@info "Generate data"
function lotka_volterra(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α*u[1] - β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  - δ*u[2]
end

tspan = (0.0f0,3.0f0)
u0 = Float32[0.44249296,4.6280594]
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka_volterra, u0,tspan, p_)
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = 0.1)

# Ideal data
tsdata = Array(solution)
# Add noise to the data
noisy_data = tsdata + Float32(1e-5)*randn(eltype(tsdata), size(tsdata))

@info "Setup neural network and auxillary functions"
ann = FastChain(FastDense(2, 32, tanh),FastDense(32, 64, tanh),FastDense(64, 32, tanh), FastDense(32, 2))
p = initial_params(ann)

function dudt_(u, p,t)
    x, y = u
    z = ann(u,p)
    [p_[1]*x + z[1],
    -p_[4]*y + z[2]]
end

prob_nn = ODEProblem(dudt_,u0, tspan, p)
s = concrete_solve(prob_nn, Tsit5(), u0, p, saveat = solution.t)

function predict(θ)
    Array(concrete_solve(prob_nn, Vern7(), u0, θ, saveat = solution.t,
                         abstol=1e-6, reltol=1e-6))
end

# No regularisation right now
function loss(θ)
    pred = predict(θ)
    sum(abs2, noisy_data .- pred), pred # + 1e-5*sum(sum.(abs, params(ann)))
end

callback(θ,l,pred) = begin
    push!(losses, l)
    if length(losses)%10 == 0
        println("Loss after $(length(losses)) iterations $(losses[end])")
    end
    false
end
@info "Train neural network"
res1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.01), cb=callback, maxiters = 100)
@info "Finished initial training with loss $(losses[end])"
res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 10000)
@info "Finished extended training with loss $(losses[end])"

# Necessary for result
@test losses[end] <= 1e-3

# Plot the data and the approximation
NNsolution = predict(res2.minimizer)

@test norm(NNsolution .- solution, 2) < 0.1

Y = ann(noisy_data, res2.minimizer)
X = noisy_data
# Create a Basis
@variables u[1:2]
# Lots of polynomials
polys = Operation[1]
for i ∈ 1:5
    push!(polys, u[1]^i)
    push!(polys, u[2]^i)
    for j ∈ i:5
        if i != j
            push!(polys, (u[1]^i)*(u[2]^j))
            push!(polys, u[2]^i*u[1]^i)
        end
    end
end

# And some other stuff
h = [cos.(u)...; sin.(u)...; polys...]
basis = Basis(h, u)

# Create an optimizer for the SINDY problem
opt = SR3()
# Create the thresholds which should be used in the search process
λ = exp10.(-10:0.05:-0.5)
# Target function to choose the results from; x = L0 of coefficients and L2-Error of the model
function eval_target(x)
    y = similar(x)
    if iszero(x[1])
        y[1] = convert(eltype(x), Inf)
    end
    y[2] = x[2]
    return y
end

alg = GoalProgramming(x->norm(x, 2), eval_target)
@info "Start sindy regression with unknown threshold"
# Test on uode derivative data
Ψ = SInDy(X[:, 2:end], Y[:, 2:end], basis, λ,  opt = opt, maxiter = 10000, normalize = true, denoise = true, alg = alg) # Succeed
@test Ψ.sparsity ≈ ones(2)
p̂ = parameters(Ψ)
@info "Build initial guess system"
# The parameters are a bit off, so we reiterate another sindy term to get closer to the ground truth
# Create function
unknown_sys = ODESystem(Ψ)
unknown_eq = ODEFunction(unknown_sys)
# Just the equations
b = Basis((u, p, t)->unknown_eq(u, [1.; 1.], t), u)
# Test on uode derivative data
@info "Refine the guess"
Ψ = SInDy(X[:, 2:end], Y[:, 2:end],b, opt = SR3(0.01), maxiter = 100, convergence_error = 1e-18) # Succeed
p̂ = parameters(Ψ)
@test isapprox(abs.(p̂), p_[2:3], atol = 3e-2)

# The parameters are a bit off, so we reiterate another sindy term to get closer to the ground truth
# Create function
unknown_sys = ODESystem(Ψ)
unknown_eq = ODEFunction(unknown_sys)


# Build a ODE for the estimated system
function approx(du, u, p, t)
    # Add SInDy Term
    α, δ, β, γ = p
    z = unknown_eq(u, [β; γ], t)
    du[1] = α*u[1] + z[1]
    du[2] = -δ*u[2] + z[2]
end
@info "Simulate system"
# Create the approximated problem and solution
ps = [p_[[1,4]]; p̂]
a_prob = ODEProblem(approx, u0, tspan, ps)
a_solution = solve(a_prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = solution.t)
@test norm(a_solution-solution, 2) < 5e-1
@test norm(a_solution-solution, Inf) < 1.5e-1

@info "Finished"
