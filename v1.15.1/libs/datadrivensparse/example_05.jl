# # [Implicit Nonlinear Dynamics : Cartpole](@id cartpole)
#
# The following is another example on how to use the [`ImplicitOptimizer`](@ref) that is taken from the [original paper](https://royalsocietypublishing.org/doi/10.1098/rspa.2020.0279).
# As always, we start by creating a corresponding dataset:

using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra
using DataDrivenSparse
#md using Plots
#md gr()

function cart_pole(u, p, t)
    du = similar(u)
    F = -0.2 + 0.5 * sin(6 * t) # the input
    du[1] = u[3]
    du[2] = u[4]
    du[3] = -(19.62 * sin(u[1]) + sin(u[1]) * cos(u[1]) * u[3]^2 + F * cos(u[1])) /
        (2 - cos(u[1])^2)
    du[4] = -(sin(u[1]) * u[3]^2 + 9.81 * sin(u[1]) * cos(u[1]) + F) / (2 - cos(u[1])^2)
    return du
end

u0 = [0.3; 0; 1.0; 0]
tspan = (0.0, 5.0)
dt = 0.05
cart_pole_prob = ODEProblem(cart_pole, u0, tspan)
solution = solve(cart_pole_prob, Tsit5(), saveat = dt)

X = solution[:, :]
DX = similar(X)
for (i, xi) in enumerate(eachcol(X))
    DX[:, i] = cart_pole(xi, [], solution.t[i])
end
t = solution.t

ddprob = ContinuousDataDrivenProblem(
    X, t, DX = DX[3:4, :],
    U = (u, p, t) -> [-0.2 + 0.5 * sin(6 * t)]
)

#md plot(ddprob)

# Note that we just included the third and forth time derivative, assuming that we already know that the velocity `x[3:4]` is equal to the time
# derivative of the position `x[1:2]`.
# Next, we define a sufficient [`Basis`](@ref). Again, we need to include `implicits` in the definition of
# our candidate functions and inform the [`Basis`](@ref) of it.

@parameters t
@variables u[1:4] du[1:2] x[1:1]
u, du, x = map(collect, [u, du, x])

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

basis = Basis(implicits, u, implicits = du, controls = x, iv = t);
#md println(basis) # hide

# We solve the problem by varying over a sufficient set of thresholds for the associated optimizer.

λ = [
    1.0e-4; 5.0e-4; 1.0e-3; 2.0e-3; 3.0e-3; 4.0e-3; 5.0e-3; 6.0e-3; 7.0e-3; 8.0e-3; 9.0e-3; 1.0e-2; 2.0e-2; 3.0e-2;
    4.0e-2; 5.0e-2
]

opt = ImplicitOptimizer(λ)
res = solve(ddprob, basis, opt)
#md println(res) # hide

# And have a look at the equations
system = get_basis(res)
#md println(system) # hide

# We have recovered the correct equations of motion!
# Another visual check using the problem and the result yields

#md plot(
#md     plot(ddprob), plot(res), layout = (1,2)
#md )

#md # ## [Copy-Pasteable Code](@id cartpole_copy_paste)
#md #
#md # ```julia
#md # @__CODE__
#md # ```

## Test #src
