using SciMLTesting
using DataDrivenLux
using JET
using Test

run_qa(
    DataDrivenLux;
    explicit_imports = true,
    # The umbrella `using DataDrivenDiffEq` pulls its public surface in implicitly;
    # making every name explicit is a large refactor tracked separately.
    ei_broken = (:no_implicit_imports,)
)
