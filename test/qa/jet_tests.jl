using DataDrivenDiffEq
using JET
using Test

# Note: JET analysis on symbolic packages can be slow and may report many
# false positives from the underlying symbolic infrastructure (Symbolics.jl,
# ModelingToolkit.jl). This test file focuses on core DataDrivenDiffEq
# functionality with concrete types to catch actual type stability issues.
#
# We use @test_opt with target_modules to only check DataDrivenDiffEq code,
# and we use broken=true for tests that detect expected polymorphic behavior
# (like problem type dispatch) rather than actual bugs.

@testset "JET Static Analysis" begin
    @testset "Basis generator type stability" begin
        # Test basis generators with concrete types
        # These should be fully type-stable as they don't involve symbolic computation
        @test_opt target_modules = (DataDrivenDiffEq,) DataDrivenDiffEq.polynomial_basis(
            2, 3
        )
        @test_opt target_modules = (DataDrivenDiffEq,) DataDrivenDiffEq.monomial_basis(2, 3)
        @test_opt target_modules = (DataDrivenDiffEq,) DataDrivenDiffEq.chebyshev_basis(2, 3)
        @test_opt target_modules = (DataDrivenDiffEq,) DataDrivenDiffEq.sin_basis(2, 3)
        @test_opt target_modules = (DataDrivenDiffEq,) DataDrivenDiffEq.cos_basis(2, 3)
        @test_opt target_modules = (DataDrivenDiffEq,) DataDrivenDiffEq.fourier_basis(2, 3)
    end

    @testset "Problem accessor type stability" begin
        X = rand(2, 10)
        t = collect(1.0:10.0)
        DX = rand(2, 10)

        prob = ContinuousDataDrivenProblem(X, t, DX)

        # Test simple accessor functions that should be type-stable
        # These accessors only check type parameters, not field values
        @test_opt target_modules = (DataDrivenDiffEq,) DataDrivenDiffEq.is_autonomous(prob)
        @test_opt target_modules = (DataDrivenDiffEq,) DataDrivenDiffEq.is_discrete(prob)
        @test_opt target_modules = (DataDrivenDiffEq,) DataDrivenDiffEq.is_continuous(prob)
        @test_opt target_modules = (DataDrivenDiffEq,) DataDrivenDiffEq.is_direct(prob)
        # Note: has_timepoints checks isempty on AbstractVector field, which has
        # expected runtime dispatch due to the abstract field type design choice
    end

    @testset "Internal constructor type stability" begin
        # Test the internal type-stable constructor directly
        X = rand(2, 10)
        t = collect(1.0:10.0)
        DX = rand(2, 10)
        Y = Matrix{Float64}(undef, 0, 0)
        U = Matrix{Float64}(undef, 0, 0)
        p = Float64[]

        # The internal constructor has runtime dispatch in _promote on Julia lts
        # due to broadcasting with abstract element types. Fixed in Julia 1.11+.
        @test_opt broken = (VERSION < v"1.11") target_modules = (DataDrivenDiffEq,) DataDrivenDiffEq._construct_datadrivenproblem(
            Val(false),
            Val(DataDrivenDiffEq.DDProbType(3)),
            Float64,
            X,
            t,
            DX,
            Y,
            U,
            p,
            :test
        )
    end
end
