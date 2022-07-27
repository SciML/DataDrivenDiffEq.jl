using DataDrivenDiffEq
using LinearAlgebra 

t = collect(-2:0.01:2)
U = [cos.(t).*exp.(-t.^2) sin.(2*t)]
S = Diagonal([2.; 3.])
V = [sin.(t).*exp.(-t) cos.(t)]
A = U*S*V'
σ = 0.5
Â = A + σ*randn(401, 401)
n_1 = norm(A-Â)
B = optimal_shrinkage(Â)
optimal_shrinkage!(Â)
@test norm(A-Â) < n_1
@test norm(A-B) == norm(A-Â)
