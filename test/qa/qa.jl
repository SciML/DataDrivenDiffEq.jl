using SciMLTesting
using DataDrivenDiffEq
using Test

# Aqua + ExplicitImports via run_qa. JET is handled separately by jet_tests.jl:
# `JET.test_package` on this package's whole method table reports many false
# positives from the re-exported symbolic infrastructure (Symbolics/ModelingToolkit),
# so the root keeps the curated `@test_opt` checks (jet_tests.jl) targeted at concrete
# DataDrivenDiffEq code instead.
run_qa(
    DataDrivenDiffEq;
    jet = false,
    explicit_imports = true,
    # The `@reexport using ModelingToolkit/StatsBase/DataInterpolations/MLUtils/CommonSolve`
    # surface is pulled in implicitly by design (this package re-exports it); making every
    # name explicit is a large refactor tracked separately.
    ei_broken = (:no_implicit_imports,)
)
