using Revise
using DataDrivenDiffEq
using DataDrivenDMD
using LinearAlgebra

A = [0.9 -0.2; 0.0 0.2]
y = [[10.0; -10.0]]
for i in 1:100
    push!(y, A * y[end])
end
X = hcat(y...)
prob = DiscreteDataDrivenProblem(X, t = 1:101)
prob = DiscreteDataDrivenProblem(X .+ 0.01 * randn(size(X)), t = 1:101)
basis = DataDrivenDiffEq.unit_basis(prob)
estimator = solve(prob, basis, DMDPINV(), options = DataDrivenCommonOptions(digits = 2))
estimator = solve(prob, DMDSVD(), options = DataDrivenCommonOptions(digits = 2))
estimator = solve(prob, TOTALDMD(), options = DataDrivenCommonOptions(digits = 2))

# works
#X = rand(8,366)
#t = collect(1.0:366.0)
#data_prob = DiscreteDataDrivenProblem(X, t)
#basis = DataDrivenDiffEq.unit_basis(data_prob)
#sol = solve(data_prob, basis, DMDPINV(), options = DataDrivenCommonOptions(digits = 2))
#print(sol)

# Use a good system 
A = diagm(-rand(50))
x0 = [10 * randn(50)]
t = [0.0]
for i in 1:200
    push!(x0, exp(A * t[end]) * x0[1])
    push!(t, t[end] + 0.01)
end
X = hcat(x0...)
t
data_prob = ContinuousDataDrivenProblem(X, t, GaussianKernel())
using Plots
plot(data_prob, legend = false)

jsol = solve(data_prob, DMDSVD(), options = DataDrivenCommonOptions(digits = 2))
plot(jsol)
print(get_basis(jsol))
reshape(get_parameter_values(get_basis(jsol)), 50, 50)
