using DataDrivenDiffEq
using DataDrivenDiffEq: collocate_data
using LinearAlgebra

@testset "Optimal Shrinkage" begin
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
end

@testset "Collocation" begin
    x = 0:0.1:10.0
    y = permutedims(x)
    z = ones(1, length(x))
    # This list does not cover all kernels since some
    # are singular
    for m in [
            EpanechnikovKernel(),
            UniformKernel(),
            TriangularKernel(),
            GaussianKernel(),
            LogisticKernel(),
            SigmoidKernel(),
            SilvermanKernel(),
        ]
        ẑ, ŷ, x̂ = collocate_data(y, x, m)
        @test ẑ ≈ z atol = 1.0e-1 rtol = 1.0e-1
        @test ŷ ≈ y atol = 1.0e-1 rtol = 1.0e-1
        @test x̂ ≈ x atol = 1.0e-1 rtol = 1.0e-1
    end

    x = 0:0.1:10.0
    y = permutedims(sin.(x))
    z = permutedims(cos.(x))

    for m in InterpolationMethod.(
            [
                LinearInterpolation,
                QuadraticInterpolation,
                CubicSpline,
            ]
        )
        ẑ, ŷ, x̂ = collocate_data(y, x, m)
        @test ẑ ≈ z atol = 1.0e-1 rtol = 1.0e-1
        @test ŷ ≈ y atol = 1.0e-1 rtol = 1.0e-1
        @test x̂ ≈ x atol = 1.0e-1 rtol = 1.0e-1
    end
end
