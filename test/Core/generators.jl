using DataDrivenDiffEq
using ModelingToolkit

@variables u[1:3]
u = collect(u)

@test isequal(sin_basis(u, 1), sin.(1 .* u))
@test isequal(cos_basis(u, 2), vcat(cos.(1 .* u), cos.(2 .* u)))
@test isequal(chebyshev_basis(u, 1), cos.(1 .* acos.(u)))
@test isequal(fourier_basis(u, 1), sin.(1 .* u ./ 2))
@test isequal(monomial_basis(u, 1), [1; u .^ 1])
@test isequal(
    polynomial_basis(u, 2),
    [
        1; u[1]^1; u[1]^2; u[2]^1; u[1]^1 * u[2]^1; u[2]^2; u[3]^1; u[1]^1 * u[3]^1;
        u[2]^1 * u[3]^1; u[3]^2
    ]
)

@test isequal(sin_basis(u, 1:2), vcat([sin.(i .* u) for i in 1:2]...))
@test isequal(cos_basis(u, 1:5), vcat([cos.(i .* u) for i in 1:5]...))
@test isequal(chebyshev_basis(u, [1; 2]), vcat([cos.(i .* acos.(u)) for i in 1:2]...))
@test isequal(fourier_basis(u, 1:2), vcat(sin.(1 .* u ./ 2), cos.(2 .* u ./ 2)))
