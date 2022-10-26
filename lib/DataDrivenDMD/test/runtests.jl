using Revise
using DataDrivenDiffEq
using DataDrivenDMD
using LinearAlgebra

A = [0.9 -0.2; 0.0 0.2]
B = [0.0; -1.0]
y = [[10.0; -10.0]]
u = [0.0]
for i in 1:1000
    push!(y, A * y[end] .+ B*u[end])
    push!(u, sin(π/4 * i))
end
X = hcat(y...)
U = permutedims(u)
prob = DiscreteDataDrivenProblem(X, t = 1:1001, U = U)
basis = DataDrivenDiffEq.unit_basis(prob)
plot(basis(prob)')
estimator = solve(prob, DMDPINV(), options = DataDrivenCommonOptions(digits = 2))
print(get_basis(estimator))
print(get_parameter_values(get_basis(estimator)))
@time estimator = solve(prob, DMDSVD(), options = DataDrivenCommonOptions(digits = 2))
estimator = solve(prob, TOTALDMD(), options = DataDrivenCommonOptions(digits = 2))
plot(estimator)
# works
#X = rand(8,366)
#t = collect(1.0:366.0)
#data_prob = DiscreteDataDrivenProblem(X, t)
#basis = DataDrivenDiffEq.unit_basis(data_prob)
#sol = solve(data_prob, basis, DMDPINV(), options = DataDrivenCommonOptions(digits = 2))
#print(sol)

# Use a good system 
A = -I + 0.001*randn(50,50)
x0 = [10 * randn(50)]
dx0 = []
t = [0.0]
for i in 1:200
    push!(dx0, A*x0[end])
    push!(x0, exp(A * t[end]) * x0[1])
    push!(t, t[end] + 0.01)
end
push!(dx0, A*x0[end])
X = hcat(x0...)
DX = hcat(dx0...)
t
data_prob = ContinuousDataDrivenProblem(X, t, DX)
using Plots
plot(data_prob, legend = false)

jsol = solve(data_prob, DMDSVD(), options = DataDrivenCommonOptions(digits = 5))
plot(jsol)
print(get_basis(jsol))
