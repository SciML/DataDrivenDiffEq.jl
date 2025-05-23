# # [Implicit Nonlinear Dynamics : Autoregulation](@id autoregulation)
#
# The following is another example on how to use the [`ImplicitOptimizer`](@ref) describing a biological autoregulation process using
# two coupled implicit equations.

using DataDrivenDiffEq
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using LinearAlgebra
using DataDrivenSparse
#md using Plots
#md gr()
using Test #src

@mtkmodel Autoregulation begin
    @parameters begin
        α = 1.0
        β = 1.3
        γ = 2.0
        δ = 0.5
    end
    @variables begin
        (x(t))[1:2] = [20.0; 12.0]
    end
    @equations begin
        D(x[1]) ~ α / (1 + x[2]) - β * x[1]
        D(x[2]) ~ γ / (1 + x[1]) - δ * x[2]
    end
end

@mtkbuild sys = Autoregulation()
tspan = (0.0, 5.0)
de_problem = ODEProblem{true, SciMLBase.NoSpecialize}(sys, [], tspan, [])
de_solution = solve(de_problem, Tsit5(), saveat = 0.005);
#md plot(de_solution)

# As always, we start by defining a [DataDrivenProblem](@ref problem) and a sufficient basis for sparse regression.

dd_prob = DataDrivenProblem(de_solution)

x = sys.x
eqs = [polynomial_basis(x, 4); D.(x); x .* D(x[1]); x .* D(x[2])]

basis = Basis(eqs, x, independent_variable = t, implicits = D.(x))

#md plot(dd_prob)

# Next to varying over different sparsity penalties, we also want to batch our data using the **TEXT**

sampler = DataProcessing(split = 0.8, shuffle = true, batchsize = 30)
res = solve(dd_prob, basis, ImplicitOptimizer(STLSQ(1e-2:1e-2:1.0)),
    options = DataDrivenCommonOptions(data_processing = sampler, digits = 2))
#md println(res) #hide

# And have a look at the resulting plot

#md plot(res)

# As well as the resulting equations of motion

system = get_basis(res) #hide
#md println(system) #hide

# Have a look at the parameter values

#md get_parameter_values(system)

# And the coefficient of determination of the result.

#md r2(res)

#md # ## [Copy-Pasteable Code](@id autoregulation_copy_paste)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
