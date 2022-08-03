using DataDrivenDiffEq
using ModelingToolkit

@variables u[1:3] du[1:3]

basis = Basis(du .+ u, u, implicits = du)
@test isequal(DataDrivenDiffEq.implicit_variables(basis), collect(du))
x = randn(3, 10)
dx = randn(3, 10)
res = x .+ dx
res_ = similar(res)
prob = DirectDataDrivenProblem(x, dx)
basis(res_, prob)

@test basis(prob) == res
@test res_ == res

prob = DiscreteDataDrivenProblem(x)
res = x[:, 1:(end - 1)] .+ x[:, 2:end]
res_ = similar(res)
basis(res_, prob)
@test basis(prob) == res
@test res_ == res
