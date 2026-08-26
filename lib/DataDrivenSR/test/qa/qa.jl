using SciMLTesting
using DataDrivenSR
using JET
using Test

# The SymbolicRegression entry points DataDrivenSR reexports so that `using DataDrivenSR`
# on its own is enough to build the `eq_options` of an `EQSearch`. Owned and documented
# upstream; kept in sync with the reexport `export` in src/DataDrivenSR.jl.
const REEXPORTS = (:SymbolicRegression, :Options)

run_qa(
    DataDrivenSR;
    reexports_allow = REEXPORTS,
    api_docs_kwargs = (; ignore = REEXPORTS, rendered_ignore = REEXPORTS),
)

@testset "Reexport surface" begin
    # Every approved reexport must actually be reachable from `using DataDrivenSR`, so the
    # allow-list cannot drift into approving names the package no longer provides.
    @testset "$name" for name in REEXPORTS
        @test name in names(DataDrivenSR)
        @test isdefined(@__MODULE__, name)
    end
end
