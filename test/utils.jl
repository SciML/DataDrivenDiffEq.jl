@info "Starting utilities tests"
@testset "Utilities" begin
    t = collect(-2:0.01:2)
    U = [cos.(t) .* exp.(-t .^ 2) sin.(2 * t)]
    S = Diagonal([2.0; 3.0])
    V = [sin.(t) .* exp.(-t) cos.(t)]
    A = U * S * V'
    σ = 0.5
    Â = A + σ * randn(401, 401)
    n_1 = norm(A - Â)
    B = optimal_shrinkage(Â)
    optimal_shrinkage!(Â)
    @test norm(A - Â) < n_1
    @test norm(A - B) == norm(A - Â)

    X = randn(3, 100)
    Y = randn(3, 100)
    k = 3

    @test AIC(k, X, Y) == 2 * k - 2 * log(sum(abs2, X - Y))
    @test AICC(k, X, Y) == AIC(k, X, Y) + 2 * (k + 1) * (k + 2) / (size(X)[2] - k - 2)
    @test BIC(k, X, Y) == -2 * log(sum(abs2, X - Y)) + k * log(size(X)[2])
    @test AICC(k, X, Y, likelihood = (X, Y) -> sum(abs, X - Y)) ==
          AIC(k, X, Y, likelihood = (X, Y) -> sum(abs, X - Y)) +
          2 * (k + 1) * (k + 2) / (size(X)[2] - k - 2)

end
