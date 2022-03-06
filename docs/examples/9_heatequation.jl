# # [PDE Discovery : Heat equations](@id heat_equation)
#
# Similar to the discovery of ODE and DAE systems, sparse regression can be used to discover the underlying 
# equations for partial differential equations as well as proposed by [PDEFind](https://www.science.org/doi/10.1126/sciadv.1602614). The following example shows how this 
# can be achieved with DataDrivenDiffEq.jl, using the heat equation with Dirichlet boundary conditions
# with the analytical soultion $u(x,t) = sin(2\pi x) exp^{-(2\pi t)^2t}$.
# We start by defining the system and generate some data.

using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using OrdinaryDiffEq
using DiffEqOperators
#md using Plots

u_analytic(x, t) = sin(2*π*x) * exp(-t*(2*π)^2)
nknots = 100 
h = 1.0/(nknots+1) 
knots = range(h, step=h, length=nknots) 
ord_deriv = 2 
ord_approx = 2 

const bc = Dirichlet0BC(Float64) 
const Δ = CenteredDifference(ord_deriv, ord_approx, h, nknots) 

t0 = 0.0 
t1 = 1.0 
u0 = u_analytic.(knots, t0)

step(u,p,t) = Δ*bc*u 
prob = ODEProblem(step, u0, (t0, t1)) 
alg = KenCarp4() 
de_solution = solve(prob, alg)

#md plot(de_solution, legend = nothing)

# Using DiffEqOperators.jl, we can define the difference operators up to order $n =4$ and vectorize the result. 

∂U = reduce(vcat, map(1:4) do n 
    δ = CenteredDifference(n, ord_approx, h, nknots)
    reshape(
        δ*bc*Array(de_solution), 1, prod(size(de_solution))
        )
    end)

# Next we collect the discretized data samples, their time derivatives and define a DataDrivenProblem

U = reshape(Array(de_solution), 1, prod(size(de_solution)))
∂ₜU = reshape(Array(de_solution(de_solution.t, Val{1})), 1, prod(size(de_solution)))

problem = DataDrivenProblem(
    U, DX = ∂ₜU, U = ∂U
)

# We choose to model the spatial derivatives $\frac{d^n u}{dx^n}$ as exegenous signals (controls), which are directly substituted into the 
# Basis defined in the following 

@parameters t x
@variables u(x, t)
∂u = map(1:4) do n 
    d = Differential(x)^n
    d(u)
end

basis = Basis([monomial_basis([u], 5);∂u], [u], independent_variable = t, controls = ∂u);
println(basis) #hide

# Afterwards, we define a sampler for the available data which performs a 80-20 train-test split and partions the training 
# data into 10 batches and solve the sparse regression using STLSQ.

sampler = DataSampler(Split(ratio = 0.8), Batcher(n = 10))
solution = solve(problem, basis, STLSQ(1e-2:1e-2:5e-1), sampler = sampler, by = :best)

# As we can see, the heat equation is recovered correclty
result(solution)
println(result(solution)) #hide

#md # ## [Copy-Pasteable Code](@id cartpole_copy_paste)
#md #
#md # ```julia
#md # @__CODE__
#md # ```

## Test #src
for r_ in [solution] #src
    @test all(l2error(r_) .< 0.5) #src
    @test all(aic(r_) .> 1e3) #src
    @test all(determination(r_) .>= 0.9) #src
end #src


