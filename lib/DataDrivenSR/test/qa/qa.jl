using SciMLTesting
using DataDrivenSR
using JET
using Test

function dependency_owned_public_names(pkg::Module)
    names = Symbol[]
    for name in SciMLTesting.public_api_names(pkg)
        isdefined(pkg, name) || continue
        value = getfield(pkg, name)
        owner = value isa Module ? value : parentmodule(value)
        owner === pkg || push!(names, name)
    end
    return Tuple(names)
end

shared_docs_src(pkg::Module) = normpath(joinpath(pkgdir(pkg), "..", "..", "docs", "src"))

run_qa(
    DataDrivenSR;
    explicit_imports = true,
    api_docs_kwargs = (;
        rendered = true,
        docs_src = shared_docs_src(DataDrivenSR),
        ignore = dependency_owned_public_names(DataDrivenSR),
        rendered_ignore = dependency_owned_public_names(DataDrivenSR),
    ),
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
    # The umbrella `using DataDrivenDiffEq` and `using DataDrivenDiffEq.<submodule>`
    # re-exports pull those public surfaces in implicitly; making every name explicit
    # is a large refactor tracked separately.
    ei_broken = (:no_implicit_imports,)
)
