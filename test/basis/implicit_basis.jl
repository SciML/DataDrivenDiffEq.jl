using Revise
using DataDrivenDiffEq
using DataDrivenDiffEq.ModelingToolkit
using LinearAlgebra

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


@variables u[1:3] du[1:3]
@parameters p[1:3]
u = collect(u)
du = collect(du)
p = collect(p)

basis = Basis([u; sin.(p .* u)], u, parameters = p)
Ξ = zeros(Float32, 2, 6)
Ξ[1,1] = -2
Ξ[1, 6] = 1
Ξ[2,3] = 1
Ξ[2,4] = 3

x = randn(3, 10)
dx = randn(3, 10)
t = 1:1:10.0
res = x .+ dx
direct_prob = DirectDataDrivenProblem(x, dx)
cont_prob = ContinuousDataDrivenProblem(x, t, DX = dx)
discrete_prob = DiscreteDataDrivenProblem(x)

d = Difference(get_iv(basis), dt = 1.0)
∂ = Differential(get_iv(basis))

direct_res = DataDrivenDiffEq.__construct_basis(Ξ, basis, direct_prob, DataDrivenDiffEq.DataDrivenCommonOptions())
discrete_res = DataDrivenDiffEq.__construct_basis(Ξ, basis, discrete_prob, DataDrivenDiffEq.DataDrivenCommonOptions())
cont_res = DataDrivenDiffEq.__construct_basis(Ξ, basis, cont_prob, DataDrivenDiffEq.DataDrivenCommonOptions())

rhs = Num.(Ξ * map(eq->eq.rhs, equations(basis)))
lhs = Num.(map(eq->eq.lhs, equations(direct_res)))
#@test 

all(isequal.(equations(direct_res), collect(lhs .~ rhs)))

@test all(isequal.(equations(imp_basis), collect(du .~ -u)))
@test all(isequal.(equations(imp_basis), collect(du .~ -u)))




basis = Basis(du .+ u, u, implicits = du)
K = Float32[0 3 0; 2 0 1; 0 0 0.5]
imp_basis = DataDrivenDiffEq.__construct_basis(K, basis, prob, DataDrivenDiffEq.DataDrivenCommonOptions())
@test all(isequal.(equations(imp_basis), collect(du .~ -u)))

