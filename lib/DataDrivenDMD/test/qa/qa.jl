using SciMLTesting
using DataDrivenDMD
using JET
using Test

run_qa(
    DataDrivenDMD;
    explicit_imports = true,
    # The umbrella `using DataDrivenDiffEq` (plus the `using DataDrivenDiffEq.<submodule>`
    # re-exports) pulls the DataDrivenDiffEq public surface in implicitly; making every
    # name explicit is a large refactor tracked separately.
    ei_broken = (:no_implicit_imports,)
)
