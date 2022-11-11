using Revise
using DataDrivenDiffEq
using DataDrivenSR
using SymbolicRegression

# Generate a multivariate function for OccamNet
X = rand(2,20)
f(x) = [sin(x[1]^2); exp(x[2])]
Y = hcat(map(f, eachcol(X))...)
# Define the options
opts = DataDrivenSR.EQSearch(eq_options = Options(unary_operators = [sin, exp], binary_operators = [+], maxdepth = 1))

@variables x y

b = Basis([x; y; x^2], [x;y])
# Define the problem
prob = ContinuousDataDrivenProblem(X, collect(1.0:1.0:20.0), Y)
# Solve the problem
res = solve(prob, b, opts)

println(res.basis)
