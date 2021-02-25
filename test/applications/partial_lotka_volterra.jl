using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra
@info "Loading DiffEqFlux"
using DiffEqFlux
@info "Loading OrdinaryDiffEq"
using OrdinaryDiffEq
using Test
using JLD2
@info "Finished loading packages"

@info "Started Lotka Volterra UODE Testset"

@info "Loading pretrained parameters"
results = jldopen(joinpath(dirname(@__FILE__), "partial_lotka_volterra.jld2"), "r")
rbf(x) = exp.(-x.^2)

U = FastChain(
    FastDense(2,5,rbf), FastDense(5,5, rbf), FastDense(5,5, rbf), FastDense(5,2)
)

# Define necessary stuff
X = results["X"]
t = results["t"]

p_trained = results["trained_parameters"]
# Plot the data and the approximation
p_ = Float32[1.3, 0.9, 0.8, 1.8]

function ude_dynamics!(du,u, p, t, p_true)
    û = U(u, p) # Network prediction
    du[1] = p_true[1]*u[1] + û[1]
    du[2] = -p_true[4]*u[2] + û[2]
end

# Closure with the known parameter
nn_dynamics!(du,u,p,t) = ude_dynamics!(du,u,p,t,p_)
# Define the problem
prob_nn = ODEProblem(nn_dynamics!,X[:, 1], (t[1], t[end]), p_trained)

function predict(θ, X = X[:,1], T = t)
    Array(solve(prob_nn, Vern7(), u0 = X, p=θ,
                tspan = (T[1], T[end]), saveat = T,
                abstol=1e-6, reltol=1e-6,
                ))
end

X̂ = predict(p_trained)
Y = U(X̂, p_trained)

# Create a Basis
@variables u[1:2]
# Lots of polynomials
polys = polynomial_basis(u, 5)

# And some other stuff
h = [sin.(u)...; polys...]
basis = Basis(h, u)

opt = STLSQ()
# Create the thresholds which should be used in the search process
λ = Float32.(exp10.(-7:0.1:0))
# Target function to choose the results from; x = L0 of coefficients and L2-Error of the model
g(x) = x[1] < 1 ? Inf : norm(x, 2)

# Test on uode derivative data
println("SINDy on learned, partial, available data")
Ψ = SINDy(X̂, Y, basis, λ,  opt, g = g, maxiter = 500, normalize = false, denoise = false, convergence_error = Float32(1e-10)) # Succeed

p̂ = parameters(Ψ)

@info "Checking equations"
found_basis = map(x->simplify(x.rhs), equations(Ψ.equations))
pps = parameters(Ψ.equations)
expected_eqs = Vector{Any}()
for _p in pps
    push!(expected_eqs, _p*u[1]*u[2])
end

@test all(isequal.(found_basis, expected_eqs))
@test isapprox(abs.(p̂), p_[2:3], atol = 1.3e-1)

# The parameters are a bit off, so we reiterate another SINDy term to get closer to the ground truth
# Create function
unknown_sys = ODESystem(Ψ)
unknown_eq = ODEFunction(unknown_sys)

# Build a ODE for the estimated system
function approx(du, u, p, t)
    # Add SINDy Term
    α, δ, β, γ = p
    z = unknown_eq(u, [β; γ], t)
    du[1] = α*u[1] + z[1]
    du[2] = -δ*u[2] + z[2]
end

@info "Simulate system"
# Create the approximated problem and solution
ps = [p_[[1,4]]; p̂]
a_prob = ODEProblem(approx, X[:,1], (t[1], t[end]), ps)
a_solution = solve(a_prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = t)
@test norm(a_solution-X, 2) < 6e-1
@test norm(a_solution-X, Inf) < 3e-1

@info "Finished"
