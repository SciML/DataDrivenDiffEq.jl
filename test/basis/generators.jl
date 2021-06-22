using Symbolics: scalarize
@variables u[1:3]
sisequal(b1, b2) = all(isequal.(scalarize(b1), b2 isa Array ? scalarize.(b2) : scalarize(b2)))
@test sisequal(sin_basis(u, 1), sin.(1 .* u))
@test sisequal(cos_basis(u, 2), vcat(cos.(1 .* u), cos.(2 .*u)))
@test sisequal(chebyshev_basis(u, 1), cos.( 1 .* acos.(u)))
@test sisequal(fourier_basis(u, 1), sin.(1 .* u ./2))
@test sisequal(monomial_basis(u, 1), [1; u.^1])
@test sisequal(polynomial_basis(u, 2), [1; u[1]^1; u[1]^2; u[2]^1; u[1]^1*u[2]^1; u[2]^2; u[3]^1; u[1]^1*u[3]^1; u[2]^1*u[3]^1; u[3]^2])

@test sisequal(sin_basis(u, 1:2), vcat([sin.(i .* u) for i in 1:2]...))
@test sisequal(cos_basis(u, 1:5), vcat([cos.(i .* u) for i in 1:5]...))
@test sisequal(chebyshev_basis(u, [1;2]), vcat([cos.(i .* acos.(u)) for i in 1:2]...))
@test sisequal(fourier_basis(u, 1:2), vcat(sin.(1 .* u ./2), cos.( 2 .* u ./ 2)))
