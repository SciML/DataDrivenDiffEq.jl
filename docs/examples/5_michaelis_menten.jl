# # [Implicit Nonlinear Dynamics : Michaelis Menten](@id michaelis_menten)
#
# What if you want to estimate an implicitly defined system of the form ``f(u_t, u, p, t) = 0``?
# The solution : Implicit Sparse Identification. This method was originally described in [this paper](http://ieeexplore.ieee.org/document/7809160/), and currently there exist [robust algorithms](https://royalsocietypublishing.org/doi/10.1098/rspa.2020.0279) to identify these systems.   
# We will focus on [Michaelis Menten Kinetics](https://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics). As before, we will define the [`DataDrivenProblem`](@ref) and the [`Basis`](@ref) containing possible candidate functions for our [`sparse_regression!`](@ref).
# Lets generate some data! We will use two experiments starting from different initial conditions.

using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using OrdinaryDiffEq
#md using Plots

function michaelis_menten(u, p, t)
    [0.6 - 1.5u[1]/(0.3+u[1])]
end


u0 = [0.5]

problem_1 = ODEProblem(michaelis_menten, u0, (0.0, 4.0));
solution_1 = solve(problem_1, Tsit5(), saveat = 0.1);
problem_2 = ODEProblem(michaelis_menten, 2*u0, (4.0, 8.0));
solution_2 = solve(problem_2, Tsit5(), saveat = 0.1);

# Since we have multiple trajectories at hand, we define a [`DataDrivenDataset`](@ref), which collects multiple problems but handles them as a unit
# for the processing.

function michaelis_menten(X::AbstractMatrix, p, t::AbstractVector)
    reduce(hcat, map((x,ti)->michaelis_menten(x, p, ti), eachcol(X), t))
end

data = (
    Experiment_1 = (X = Array(solution_1), t = solution_1.t, DX = michaelis_menten(Array(solution_1),[], solution_1.t) ), 
    Experiment_2 = (X = Array(solution_2), t = solution_2.t, DX = michaelis_menten(Array(solution_2),[], solution_2.t))
)

prob = DataDrivenDiffEq.ContinuousDataset(data);
#md plot(prob)

# Next, we define our [`Basis`](@ref). Since we want to identify an implicit system, we have to include  
# some candidate terms which use these as an argument and inform our constructor about the meaning of these variables.

@parameters t
D = Differential(t)
@variables u[1:1](t)
h = [monomial_basis(u[1:1], 4)...]
basis = Basis([h; h .* D(u[1])], u, implicits = D.(u), iv = t)
println(basis) # hide
    
    
# Next, we define the [`ImplicitOptimizer`](@ref) and `solve` the problem. It wraps a standard optimizer, by default [`STLSQ`](@ref), and performs 
# implicit sparse regression upon the selected basis.
# To improve our result, we batch the data by using a [`DataSampler`](@ref). Here, we use a train-test split of 0.8 and 
# divide the training data into 10 batches. Since we are using a batching process, we can also use a different 
    
sampler = DataSampler(
    Split(ratio = 0.8), Batcher(n = 10)
)

opt = ImplicitOptimizer(1e-1:1e-1:5e-1)
res = solve(prob, basis, opt,  normalize = false, denoise = false, by = :min, sampler = sampler, maxiter = 1000);
println(res) # hide

# As we can see, the [`DataDrivenSolution`](@ref) has good metrics. Furthermore, inspection of the underlying system shows that the original equations have been recovered correctly:
    
system = result(res)
println(system) # hide

#md plot(
#md     plot(prob), plot(res), layout = (1,2)
#md )

#md # ## [Copy-Pasteable Code](@id michaelis_menten_copy_paste)
#md #
#md # ```julia
#md # @__CODE__
#md # ```

## Test #src
for r_ in [res] #src
    @test all(l2error(r_) .< 0.01) #src
    @test all(aic(r_) .< -600.0) #src
    @test all(determination(r_) .>= 0.9) #src
end #src


