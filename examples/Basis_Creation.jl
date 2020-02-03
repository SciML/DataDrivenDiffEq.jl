using LinearAlgebra
using DataDrivenDiffEq
using Plots
using ModelingToolkit

# Frist we define a set of variables and parameters
@variables u[1:3]
@parameters w[1:2]

# Now the equations which form our basis
h = [u[1]; u[2]; cos(w[1]*u[2]+w[2]*u[3]); u[3]+u[2]]

# Then we simply create a basis
b = Basis(h, u, parameters = w)

# And look at the corresponding eqs
println(b)

# Suppose we want to add another equation, say sin(u[1])
# The basis behaves like an array
push!(b, sin(u[1]))
size(b) # (5)

# Adding an equation which is already present, does not change the basis
push!(b, sin(u[1]))
size(b) # Still 5

# We can iterate over the basis
for bi in b
    println(bi)
end

# Index specific eqs
b[3]

# And of course evaluate
# With fixed parameters
b([1;2;3], p = [2; 4])
# And without
b([1;2;3])
# Or for trajectories
X = randn(3, 40)
Y_p = b(X)
Y = b(X, p = [2;4])


# This allows you to transform a basis simply via
@variables x[1:2]
y = [sin(x[1]); cos(x[1]); x[2]]
b2 = Basis(b(y), x, parameters = w)
println(b2)


# We can merge basis
b3 = merge(b, b2)

# Also in place
merge!(b3, b2)
println(b3)

# Get the variables or parameters
variables(b)
parameters(b)

# We can also check if two bases are equal
b == b
