using SciMLTesting
using DataDrivenSparse
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
    DataDrivenSparse;
    explicit_imports = true,
    api_docs_kwargs = (;
        rendered = true,
        docs_src = shared_docs_src(DataDrivenSparse),
        ignore = dependency_owned_public_names(DataDrivenSparse),
        rendered_ignore = dependency_owned_public_names(DataDrivenSparse),
    ),
    # The `unique!`/`deleteat!` Basis methods that DataDrivenDiffEq defines on
    # `Symbolics.Num`/`Symbolics.Arr` vectors are ambiguous against `Base.unique!`/
    # `Base.deleteat!`; the ambiguity is inherited from the parent package (it predates
    # this conversion and Aqua was never run before), so it is marked broken here.
    aqua_broken = (:ambiguities,),
    ei_kwargs = (;
        # DataDrivenDiffEq exposes these names as the (non-`export`ed) extension API its
        # own sublibraries build on; declaring them `public` upstream is tracked separately.
        all_explicit_imports_are_public = (;
            ignore = (
                :AbstractDataDrivenAlgorithm, :AbstractDataDrivenResult,
                :InternalDataDrivenProblem,
            ),
        ),
        all_qualified_accesses_are_public = (;
            # __construct_basis/is_implicit: DataDrivenDiffEq internals; transform: StatsBase.
            ignore = (:__construct_basis, :is_implicit, :transform),
        ),
    ),
    # The umbrella `using DataDrivenDiffEq` (plus the `using DataDrivenDiffEq.<submodule>`
    # re-exports) pulls the DataDrivenDiffEq public surface in implicitly; making every
    # name explicit is a large refactor tracked separately.
    ei_broken = (:no_implicit_imports,)
)
