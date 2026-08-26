using SciMLTesting
using DataDrivenDiffEq
using Test

# The upstream API DataDrivenDiffEq deliberately reexports so that `using DataDrivenDiffEq`
# on its own is enough to write a `Basis`, choose a collocation method, normalize and batch
# the data, and inspect the recovered system. Owned and documented upstream; kept in sync
# with the reexport `export` blocks in src/DataDrivenDiffEq.jl.
const REEXPORTS = (
    :solve,
    # Symbolics / ModelingToolkitBase symbolic DSL
    Symbol("@variables"), Symbol("@parameters"), :Differential, :Equation, :Num,
    :build_function, :get_variables,
    # ModelingToolkitBase system accessors
    :equations, :get_iv, :get_observed, :independent_variable, :observed, :parameters,
    :unknowns,
    # StatsAPI / StatsBase statistical interface
    :aic, :aicc, :bic, :dof, :loglikelihood, :nobs, :nullloglikelihood, :r2, :rss,
    :summarystats,
    # DataInterpolations collocation methods
    :BSplineApprox, :BSplineInterpolation, :ConstantInterpolation, :CubicSpline,
    :Curvefit, :LagrangeInterpolation, :LinearInterpolation, :QuadraticInterpolation,
    :QuadraticSpline,
    # MLUtils / StatsBase data processing
    :DataLoader, :splitobs, :UnitRangeTransform, :ZScoreTransform,
)

run_qa(
    DataDrivenDiffEq;
    reexports_allow = REEXPORTS,
    api_docs_kwargs = (; ignore = REEXPORTS, rendered_ignore = REEXPORTS),
)

@testset "Reexport surface" begin
    # Every approved reexport must actually be reachable from `using DataDrivenDiffEq`, so
    # the allow-list cannot drift into approving names the package no longer provides.
    @testset "$name" for name in REEXPORTS
        @test name in names(DataDrivenDiffEq)
        @test isdefined(@__MODULE__, name)
    end
end
