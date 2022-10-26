
A = [0.9 -0.2; 0.0 0.2]
y = [[10.0; -10.0]]
for i in 1:10
    push!(y, A * y[end])
end
X = hcat(y...)
prob = DiscreteDataDrivenProblem(X, t = 1:11)
estimator = solve(prob, DMDPINV())
