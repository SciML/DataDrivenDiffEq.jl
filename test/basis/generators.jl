@variables u[1:3]
@test all(isequal.(sin_basis(u, 1), sin.(1 .* u)))
@test all(isequal.(cos_basis(u, 2), vcat(cos.(1 .* u), cos.(2 .*u))))
@test all(isequal.(chebyshev_basis(u, 1), cos.( 1 .* acos.(u))))
@test all(isequal.(fourier_basis(u, 1), sin.(1 .* u ./2)))
@test all(isequal.(monomial_basis(u, 1), [1; u.^1]))
@test all(isequal.(polynomial_basis(u, 2), [1; u[1]^1; u[1]^2; u[2]^1; u[1]^1*u[2]^1; u[2]^2; u[3]^1; u[1]^1*u[3]^1; u[2]^1*u[3]^1; u[3]^2]))

@test all(isequal.(sin_basis(u, 1:2), vcat([sin.(i .* u) for i in 1:2]...)))
@test all(isequal.(cos_basis(u, 1:5), vcat([cos.(i .* u) for i in 1:5]...)))
@test all(isequal.(chebyshev_basis(u, [1;2]), vcat([cos.(i .* acos.(u)) for i in 1:2]...)))
@test all(isequal.(fourier_basis(u, 1:2), vcat(sin.(1 .* u ./2), cos.( 2 .* u ./ 2))))
