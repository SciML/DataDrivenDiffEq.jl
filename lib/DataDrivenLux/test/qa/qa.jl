using SciMLTesting
using DataDrivenLux
using JET
using Test

run_qa(
    DataDrivenLux;
    explicit_imports = true,
    ei_kwargs = (;
        all_explicit_imports_are_public = (;
            ignore = (
                # DataDrivenDiffEq non-`export`ed extension API its sublibraries build on.
                :AbstractDataDrivenAlgorithm, :AbstractDataDrivenProblem,
                :AbstractDataDrivenResult, :InternalDataDrivenProblem,
                # External non-public names, used deliberately.
                :AbstractBackend, :ForwardDiffBackend, :gradient,  # AbstractDifferentiation
                :converged, :Options,                              # Optim
                :isempty,                                          # IntervalArithmetic
                :square,                                           # InverseFunctions
            ),
        ),
        all_explicit_imports_via_owners = (;
            ignore = (:isempty,),  # IntervalArithmetic.isempty (owner Base)
        ),
        all_qualified_accesses_are_public = (;
            ignore = (
                # DataDrivenDiffEq internals accessed qualified.
                :get_f, :get_fit_targets, :get_implicit_data, :get_oop_args,
                :remake_problem, :_set_default_val, :AbstractDataDrivenResult,
                :getdefaultval,                          # Symbolics
                :isvariable,                             # ModelingToolkit
                :promote_eltype,                         # Base
                :AbstractBackend, :ForwardDiffBackend, :gradient,  # AbstractDifferentiation
                :converged, :Options,                    # Optim
            ),
        ),
        all_qualified_accesses_via_owners = (;
            ignore = (:isvariable,),  # ModelingToolkit.isvariable (owner ModelingToolkitBase)
        ),
    ),
    # The umbrella `using DataDrivenDiffEq` pulls its public surface in implicitly;
    # making every name explicit is a large refactor tracked separately.
    ei_broken = (:no_implicit_imports,)
)
