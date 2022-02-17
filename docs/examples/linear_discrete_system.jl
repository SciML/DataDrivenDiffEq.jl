# # [Linear Time Discrete System](@id linear_discrete)
# 
# We will start by estimating the underlying dynamical system of a time discrete process based on some measurements via [Dynamic Mode Decomposition](https://arxiv.org/abs/1312.0041) on a simple linear system of the form ``u(k+1) = A u(k)``.
# 
# At first, we simulate the correspoding system using `OrdinaryDiffEq.jl` and generate a [`DiscreteDataDrivenProblem`](@ref DataDrivenProblem) from the simulated data.

#md using DataDrivenDiffEq
#md using ModelingToolkit
#md using LinearAlgebra
#md using OrdinaryDiffEq
#md using Plots

A = [0.9 -0.2; 0.0 0.2]
u0 = [10.0; -10.0]
tspan = (0.0, 11.0)

f(u,p,t) = A*u

sys = DiscreteProblem(f, u0, tspan)
sol = solve(sys, FunctionMap());

# Next we transform our simulated solution into a [`DataDrivenProblem`](@ref). Given that the solution knows its a discrete solution, we can simply write

prob = DataDrivenProblem(sol)

# And plot the solution and the problem 

#md plot(sol, label = string.([:x₁ :x₂])) 
#md scatter!(prob)

# To estimate the underlying operator in the states ``x_1, x_2``, we `solve` the estimation problem using the [`DMDSVD`](@ref) algorithm for approximating the operator. First, we will have a look at the [`DataDrivenSolution`](@ref)

res = solve(prob, DMDSVD(), digits = 1)
#md println(res) # hide

# We see that the system has been recovered correctly, indicated by the small error and high AIC score of the result. We can confirm this by looking at the resulting [`Basis`](@ref)

system = result(res)
using Symbolics

#md println(system) # hide

# And also plot the prediction of the recovered dynamics

#md plot(res)

# Or a have a look at the metrics of the result

#md metrics(res)

# To have a look at the representation of the operator as a `Matrix`, we can simply call

#md Matrix(system)

# to see that the operator is indeed our initial `A`. Since we have a linear representation, we can gain further insights into the stability of the dynamics via its eigenvalues

#md eigvals(system)

# And plot the stability margin of the discrete System

#md φ = 0:0.01π:2π 
#md plot(sin.(φ), cos.(φ), xlabel = "Real", ylabel = "Im", label = "Stability margin", color = :red, linestyle = :dash)
#md scatter!(real(eigvals(system)), imag(eigvals(system)), label = "Eigenvalues", color = :black, marker = :cross) 

# Similarly, we could use a sparse regression to derive our system from our data. We start by defining a [`Basis`](@ref)

using ModelingToolkit

@parameters t
@variables x[1:2](t)

basis = Basis(x, x, independent_variable = t, name = :LinearBasis)
#md print(basis) #hide

# Afterwards, we simply `solve` the already defined problem with our `Basis` and a `SparseOptimizer`

sparse_res = solve(prob, basis, STLSQ())
#md println(sparse_res) #hide

# Which holds the same equations
sparse_system = result(sparse_res)
#md println(sparse_system) #hide

# Again, we can have a look at the result

#md plot(
#md     plot(prob), plot(sparse_res), layout = (1,2)
#md )

# Both results can be converted into a `DiscreteProblem`

@named sys = DiscreteSystem(equations(sparse_system), get_iv(sparse_system),states(sparse_system), parameters(sparse_system))
#md println(sys) #hide

# And simulated using `OrdinaryDiffEq.jl` using the (known) initial conditions and the parameter mapping of the estimation.

x0 = [x[1] => u0[1], x[2] => u0[2]]
ps = parameter_map(sparse_res)

discrete_prob = DiscreteProblem(sys, x0, tspan, ps)
estimate = solve(discrete_prob, FunctionMap());

# And look at the result
#md plot(sol, color = :black)
#md plot!(estimate, color = :red, linestyle = :dash)

#md # ## [Copy-Pasteable Code](@id linear_discrete_copy_paste)
#md #
#md # ```julia
#md # @__CODE__
#md # ```

## Test the result #src
for r_ in [res, sparse_res] #src
    @test all(l2error(r_) .<= 1e-10) #src
    @test all(aic(r_) .>= 1e10) #src
    @test all(determination(r_) .≈ 1.0) #src
end  #src
@test Array(sol) ≈ Array(estimate) #src


