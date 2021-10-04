using Symbolics: scalarize
@variables x[1:3] u[1:2]
@parameters p[1:2] t

g = [x[1] * p[1] + p[2] * x[2]; x[2] * u[1]; u[2] * x[3] + exp(-t)]

b = Basis(g, x, parameters = p, controls = u, iv = t)

# Vector call
x0 = randn(3)
p0 = randn(2)
t0 = 0.0
u0 = randn(2)

true_res(x, p, t, u) =
    [sum(x[1:2] .* p); x[2] .* u[1]; u[2] .* x[3] .+ exp.(-t)]
true_res_(x, p, t) =
    scalarize.(hcat([true_res(x[:, i], p, t[i], zeros(2)) for i = 1:100]...))
true_res_(x, p, t, u) =
    scalarize.(hcat([true_res(x[:, i], p, t[i], u[:, i]) for i = 1:100]...))

@test isequal(b(x0), b.f(x0, p, t, zeros(2)))
@test isequal(b(x0, p0), b.f(x0, p0, t, zeros(2)))
@test isequal(b(x0, p0, t0), b.f(x0, p0, t0, zeros(2)))
@test isequal(b(x0, p0, t0, u0), b.f(x0, p0, t0, u0))

# Array call
x0 = randn(3, 100)
p0 = randn(2)
t0 = randn(100)
u0 = randn(2, 100)

# These first two fail, since exp(-t) != exp(getindex(t,1))
@test isequal(b(x0), true_res_(x0, p, [t for i = 1:100]))
@test isequal(b(x0, p0), true_res_(x0, p0, [t for i = 1:100]))
@test isequal(b(x0, p0, t0), true_res_(x0, p0, t0))
@test isequal(b(x0, p0, t0, u0), true_res_(x0, p0, t0, u0))


@parameters w[1:2] t
@variables u[1:3](t)

h = [u[1]; u[2]; cos(w[1] * u[2] + w[2] * u[3]); 5*u[3] + u[2]]
h_not_unique = [u[1]; u[1]; u[1]^1; h; 1]
basis = Basis(h_not_unique, u, parameters = w, iv = t)
basis_2 =
    Basis(h_not_unique, u, parameters = w, iv = t, linear_independent = true)
# Check getters
@test isequal(states(basis), u)
@test isequal(parameters(basis), w)
@test isequal(ModelingToolkit.get_iv(basis), t)
@test isequal(controls(basis), [])

# Check free parameter calculation
@test free_parameters(basis, operations = [+]) == 7
@test free_parameters(basis_2) == 6
@test free_parameters(basis, operations = [+, cos]) == 8
@test free_parameters(basis_2, operations = [+, cos]) == 7
@test DataDrivenDiffEq.count_operation(
    (1 + cos(u[2]) * sin(u[1]))^3,
    [+, cos, ^, *],
) == 4

# Check array functionalities
basis_2 = unique(basis)
@test isequal(basis, basis_2)
@test size(basis) == (5,)
@test basis([1.0; 2.0; π], [0.0; 1.0]) ≈ [1.0; 2.0; -1.0; 5*π + 2.0; 1.0]
@test size(basis) == size(basis_2)
push!(basis_2, sin(u[2]))
@test size(basis_2)[1] == length(h) + 2
basis_3 = merge(basis, basis_2)
@test size(basis_3) == (6,)
@test isequal(states(basis_3), states(basis_2))
@test isequal(parameters(basis_3), parameters(basis_2))
merge!(basis_3, basis)
@test basis_3 == basis_2
push!(basis, 5*u[3] + u[2])
unique!(basis) # Does not remove
@test size(basis) == (5,)

# Further callables
@variables a
g = [u[1]; u[3]; a]
basis = Basis(g, [u; a])
@test basis([1; 2; 3; 4]) == [1; 3; 4]
g = [u[1]; u[3]; u[2]]
basis = Basis(g, u, parameters = [])
X = ones(Float64, 3, 10)
X[1, :] .= 3 * X[1, :]
X[3, :] .= 5 * X[3, :]
# Check the array evaluation
@test basis(X) ≈ [1.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0] * X
Y = similar(X)
basis(Y, X)
@test Y ≈ [1.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0] * X
f = DataDrivenDiffEq.jacobian(basis)
@test f([1; 1; 1], [0.0; 0.0], 0.0) ≈ [1.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0]
@test_nowarn [xi for xi in basis]
@test_nowarn basis[2:end];

@variables u[1:3] t

g = [u[2]; -sin(u[1]) * exp(-t); u[2] + u[3]]
basis = Basis(g, u, iv = t)

f_(u, p, t) = [u[3]; u[2] * u[1]; p[1] * sin(u[1]) * u[2]; p[2] * t]
b = Basis(f_, u, parameters = w, iv = t)
@test f_([1; 2; 3], [2; 0], 3.0) ≈ b([1; 2; 3], [2; 0], 3.0)
