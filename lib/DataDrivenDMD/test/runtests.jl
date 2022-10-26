using Revise
using DataDrivenDiffEq
using DataDrivenDMD

A = [0.9 -0.2; 0.0 0.2]
y = [[10.0; -10.0]]
for i in 1:10
    push!(y, A * y[end])
end
X = hcat(y...)
prob = DiscreteDataDrivenProblem(X, t = 1:11)
prob = DiscreteDataDrivenProblem(X .+ 0.01*randn(size(X)), t = 1:11)

@btime estimator = solve(prob, DMDPINV(), options = DataDrivenCommonOptions(digits = 2))
@btime estimator = solve(prob, DMDSVD(), options = DataDrivenCommonOptions(digits = 2))
@btime estimator = solve(prob, TOTALDMD(), options = DataDrivenCommonOptions(digits = 2))

# works
X = randn(31,366)
t = collect(1.0:366.0)
data_prob = ContinuousDataDrivenProblem(X, t)
sol = solve(data_prob, DMDSVD(), options = DataDrivenCommonOptions(digits = 2))
# does not work
X = randn(32,366)
t = collect(1.0:366.0)
data_prob = ContinuousDataDrivenProblem(X, t)
sol = solve(data_prob, DMDSVD())