using DataDrivenDiffEq
using DataDrivenSparse
using OrdinaryDiffEq
using Test
using StatsBase

function cart_pole(u, p, t)
    du = similar(u)
    F = -0.2 + 0.5 * sin(6 * t) # the input
    du[1] = u[3]
    du[2] = u[4]
    du[3] = -(19.62 * sin(u[1]) + sin(u[1]) * cos(u[1]) * u[3]^2 + F * cos(u[1])) / (2 - cos(u[1])^2)
    du[4] = -(sin(u[1]) * u[3]^2 + 9.81 * sin(u[1]) * cos(u[1]) + F) / (2 - cos(u[1])^2)
    return du
end

u0 = [0.3; 0; 1.0; 0]
tspan = (0.0, 5.0)
dt = 0.05
cart_pole_prob = ODEProblem(cart_pole, u0, tspan)
solution = solve(cart_pole_prob, Tsit5(), saveat = dt)

# Create the differential data
X = Array(solution)
DX = similar(X)
for (i, xi) in enumerate(eachcol(X))
    DX[:, i] = cart_pole(xi, [], solution.t[i])
end
t = solution.t

ddprob = ContinuousDataDrivenProblem(
    X, t, DX = DX[3:4, :],
    U = (u, p, t) -> [-0.2 + 0.5 * sin(6 * t)]
)

@variables u[1:4] x[1:1] t
du = [Symbolics.variable("du", i) for i in 3:4]
u = collect(u)
du = collect(du)
x = collect(x)

polys = polynomial_basis(u, 2)
push!(polys, sin.(u[1]))
push!(polys, cos.(u[1]))
push!(polys, sin.(u[1])^2)
push!(polys, cos.(u[1])^2)
push!(polys, sin.(u[1]) .* u[3:4]...)
push!(polys, sin.(u[1]) .* u[3:4] .^ 2...)
push!(polys, sin.(u[1]) .* cos.(u[1])...)
push!(polys, sin.(u[1]) .* cos.(u[1]) .* u[3:4]...)
push!(polys, sin.(u[1]) .* cos.(u[1]) .* u[3:4] .^ 2...)
implicits = [du; du[1] .* u; du[2] .* u; du .* cos(u[1]); du .* cos(u[1])^2; polys]
push!(implicits, x...)
push!(implicits, x[1] * cos(u[1]))
push!(implicits, x[1] * sin(u[1]))

basis = Basis(implicits, u, controls = x, iv = t, implicits = du)

λ = [
    1.0e-4; 5.0e-4; 1.0e-3; 2.0e-3; 3.0e-3; 4.0e-3; 5.0e-3; 6.0e-3; 7.0e-3; 8.0e-3; 9.0e-3; 1.0e-2; 2.0e-2; 3.0e-2;
    4.0e-2; 5.0e-2
]

res = solve(
    ddprob, basis, ImplicitOptimizer(λ),
    options = DataDrivenCommonOptions(verbose = false, digits = 3)
)
@test r2(res) >= 0.95
@test rss(res) <= 1.0e-2
@test dof(res) == 10
@test get_parameter_values(res.basis) ≈
    [-0.101, 0.05, -1.0, -0.05, -0.05, -0.203, 0.101, -0.101, -1.0, -0.101]
