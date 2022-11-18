```@meta
EditURL = "<unknown>/docs/src/libs/datadrivensparse/example_04.jl"
```

# [Implicit Nonlinear Dynamics : Autoregulation](@id autoregulation)

The following is another example on how to use the [`ImplicitOptimizer`](@ref) describing a biological autoregulation process using
two coupled implicit equations.

````@example example_04
using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra
using DataDrivenSparse
using Plots
gr()

@parameters begin
    t
    α = 1.0
    β = 1.3
    γ = 2.0
    δ = 0.5
end

@variables begin
    x[1:2](t) = [20.0; 12.0]
end

x = collect(x)
D = Differential(t)

eqs = [
    D(x[1]) ~ α/(1+x[2])-β*x[1];
    D(x[2]) ~ γ/(1+x[1])-δ*x[2];
]

sys = ODESystem(eqs, t, x, [α, β, γ, δ], name = :Autoregulation)

x0 = [x[1] => 20.0; x[2] => 12.0]

tspan = (0.0, 5.0)

de_problem = ODEProblem{true, SciMLBase.NoSpecialize}(sys, x0, tspan)
de_solution = solve(de_problem, Tsit5(), saveat = 0.005);
plot(de_solution)
````

As always, we start by defining a [DataDrivenProblem](@ref problem) and a sufficient basis for sparse regression.

````@example example_04
dd_prob = DataDrivenProblem(de_solution)

eqs = [
    polynomial_basis(x, 4); D.(x); x .* D(x[1]); x .* D(x[2])
    ]

basis = Basis(eqs, x, independent_variable = t, implicits = D.(x))

plot(dd_prob)
````

Next to varying over different sparsity penalties, we want also to batch our data using the **TEXT**

````@example example_04
sampler = DataProcessing(split = 0.8, shuffle = true, batchsize = 30)
res = solve(dd_prob, basis, ImplicitOptimizer(STLSQ(1e-2:1e-2:1.0)), options = DataDrivenCommonOptions(data_processing = sampler, digits = 2, ))
println(res) #hide
````

And have a look at the outcoming plots

````@example example_04
plot(res)
````

As well as the resulting equations of motion

````@example example_04
system = get_basis(res) #hide
println(system) #hide
````

Have a look at the parameter values

````@example example_04
get_parameter_values(system)
````

And the coefficient of determination

````@example example_04
r2(system)
````

## [Copy-Pasteable Code](@id autoregulation_copy_paste)

```julia
using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra
using DataDrivenSparse

@parameters begin
    t
    α = 1.0
    β = 1.3
    γ = 2.0
    δ = 0.5
end

@variables begin
    x[1:2](t) = [20.0; 12.0]
end

x = collect(x)
D = Differential(t)

eqs = [
    D(x[1]) ~ α/(1+x[2])-β*x[1];
    D(x[2]) ~ γ/(1+x[1])-δ*x[2];
]

sys = ODESystem(eqs, t, x, [α, β, γ, δ], name = :Autoregulation)

x0 = [x[1] => 20.0; x[2] => 12.0]

tspan = (0.0, 5.0)

de_problem = ODEProblem{true, SciMLBase.NoSpecialize}(sys, x0, tspan)
de_solution = solve(de_problem, Tsit5(), saveat = 0.005);

dd_prob = DataDrivenProblem(de_solution)

eqs = [
    polynomial_basis(x, 4); D.(x); x .* D(x[1]); x .* D(x[2])
    ]

basis = Basis(eqs, x, independent_variable = t, implicits = D.(x))

sampler = DataProcessing(split = 0.8, shuffle = true, batchsize = 30)
res = solve(dd_prob, basis, ImplicitOptimizer(STLSQ(1e-2:1e-2:1.0)), options = DataDrivenCommonOptions(data_processing = sampler, digits = 2, ))

system = get_basis(res) #hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

