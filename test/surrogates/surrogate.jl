#using Revise
#
#using DataDrivenDiffEq
#using ModelingToolkit
#using Random

# Init the problem
Random.seed!(1212)
ws = rand(4) .+ 0.2
f(x) = [-2.0sin(first(x))+5.0last(x); sum(ws[1:3] .* x[1:3]) + ws[end]; 3.0-2.0*x[6]/(1.0+3.0*x[5]^2) + x[3]]
x = randn(20, 200)
y = reduce(hcat, map(f, eachcol(x)))
ȳ = y .+ 1e-3*randn(size(y))

# Add a generator
generator(u) = vcat(polynomial_basis(u, 2), sin.(u))

generator(u,v) = begin
    explicits = polynomial_basis(u, 2)
    implicits = explicits .* v
    return vcat(explicits, implicits)
end

# Add the solver and the problem
sol = SurrogateSolvers(generator, STLSQ(1e-2:1e-2:1.0), ImplicitOptimizer(0.1:0.1:1.0), maxiters = 1000, normalize_coefficients = false, progress = true)
prob = DirectDataDrivenProblem(x, ȳ)
res = solve(prob, f, sol, abstol = 3e-2, reltol = 3e-2)

# Tests here
for r in res
    println(result(r))
    println(metrics(r))
    println(parameters(r))
end