using LinearAlgebra
using DataDrivenDiffEq
using Plots
using ModelingToolkit

# First, we define a set of variables and parameters
@variables u[1:3]
@parameters t w[1:2]

# Now, the equations which form our basis
h = [u[1]; u[2]; cos(w[1]*u[2]+w[2]*u[3])]

# Then, we simply create a basis
b= Basis(h, u, parameters = w)
ODESystem(b)

# Basis are callable with out of place
x = b([1;2;3])
# And in place
dx = similar(x)
b(dx, [1;2;3])
dx
# And look at the corresponding eqs
println(b)

# Suppose we want to add another equation, say sin(u[1])
# The basis behaves like an array
push!(b, sin(u[1]))
size(b) # (4)

# Adding an equation which is already present does not change the basis
push!(b, sin(u[1]))
size(b) # Still 4

# We can iterate over the basis
for bi in b
    println(bi)
end

# Index-specific eqs
b[3]

# And, of course, evaluate over trajectories
X = randn(3, 40)
Y_p = b(X)
# With parameter and time
t = independent_variable(b)
push!(b, sin(t))
Y = b(X, [2;4], 0:39)


# This allows you to transform a basis simply via
@variables x[1:2]
y = [sin(x[1]); cos(x[1]); x[2]]

b2 = Basis(b(y, parameters(b), independent_variable(b)), x, parameters = w, iv = t)
println(b2)
b2([1;2;3], [1 3], 0.0)

# We can merge basis
b3 = merge(b, b2)

# Also in place
merge!(b3, b2)
println(b3)

# Get the variables or parameters
variables(b)
parameters(b)
independent_variable(b)

# We can also check if two bases are equal
b == b

# Every function which is defined over Operations can be parsed
f(u, p, t) = [u[3]; u[2]*u[1]; sin(u[1])*u[2]]
b = Basis(f, u)

using Flux
NNlib.σ(x::Operation) = 1 / (1+exp(-x))
# Build a fully
c = Chain(Dense(3,5,σ), Dense(5, 2, σ))
ps, re = Flux.destructure(c)
@parameters p[1:length(ps)]
g(u, p, t) = re(p)(u)
b = Basis(g, u, parameters = p)
println(b)
