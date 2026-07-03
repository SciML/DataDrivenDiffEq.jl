using SciMLTesting
using DataDrivenSR
using JET
using Test

run_qa(
    DataDrivenSR;
    explicit_imports = true,
    ei_kwargs = (;
        all_explicit_imports_are_public = (;
            # DataDrivenDiffEq non-`export`ed extension API its sublibraries build on.
            ignore = (
                :AbstractDataDrivenAlgorithm, :AbstractDataDrivenResult,
                :InternalDataDrivenProblem,
            ),
        ),
        all_qualified_accesses_are_public = (;
            ignore = (
                # DataDrivenDiffEq internals accessed qualified.
                :assert_lhs, :get_implicit_data, :remake_problem, :_set_default_val,
                :setdefaultval,  # Symbolics
                :Sym,            # SymbolicUtils
                :toparam,        # ModelingToolkit
            ),
        ),
        all_qualified_accesses_via_owners = (;
            ignore = (:toparam,),  # ModelingToolkit.toparam (owner ModelingToolkitBase)
        ),
    ),
    # The umbrella `using DataDrivenDiffEq` and `@reexport using SymbolicRegression`
    # (plus the `using DataDrivenDiffEq.<submodule>` re-exports) pull those public
    # surfaces in implicitly; making every name explicit is a large refactor tracked
    # separately.
    ei_broken = (:no_implicit_imports,)
)
