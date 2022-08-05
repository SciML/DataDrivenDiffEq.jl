using Revise
using DataDrivenDiffEq
using DataDrivenDMD
using ModelingToolkit
using LinearAlgebra

@variables x[1:3] u[1:2] 
@parameters p[1:2]

k_discrete = Koopman(x, x)
k_continuous = Koopman(x, x, K = Diagonal(-0.1*ones(length(x))), is_discrete = false)

@test !DataDrivenDiffEq.is_implicit(k_discrete)
@test !DataDrivenDiffEq.is_implicit(k_continuous)
@test DataDrivenDMD.is_discrete(k_discrete)
@test DataDrivenDMD.is_continuous(k_continuous)
@test !is_stable(k_discrete)
@test is_stable(k_continuous)

@test_throws AssertionError frequencies(k_discrete)
@test_throws AssertionError modes(k_discrete)
@test_throws AssertionError generator(k_discrete)
@test_throws AssertionError operator(k_continuous)


@test operator(k_discrete) ≈ I
@test generator(k_continuous) ≈ Diagonal(-0.1*ones(length(x)))

@test outputmap(k_discrete) ≈ I
@test outputmap(k_continuous) ≈ I



A = [0.9 -0.2; 0.0 0.2]
y = [[10.0; -10.0]]
for i in 1:10
    push!(y, A * y[end])
end
X = hcat(y...)
prob = DiscreteDataDrivenProblem(X, t = 1:11)
estimator = solve(prob, DMDPINV())
