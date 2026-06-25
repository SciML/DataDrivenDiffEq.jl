using SciMLTesting
using DataDrivenSR
using JET
using Test

run_qa(
    DataDrivenSR;
    explicit_imports = true,
    # The umbrella `using DataDrivenDiffEq` and `@reexport using SymbolicRegression`
    # (plus the `using DataDrivenDiffEq.<submodule>` re-exports) pull those public
    # surfaces in implicitly; making every name explicit is a large refactor tracked
    # separately.
    ei_broken = (:no_implicit_imports,)
)
