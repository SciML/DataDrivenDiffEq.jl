using SciMLTesting
using DataDrivenDMD
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
    DataDrivenDMD;
    explicit_imports = true,
    api_docs_kwargs = (;
        rendered = true,
        docs_src = shared_docs_src(DataDrivenDMD),
        ignore = dependency_owned_public_names(DataDrivenDMD),
        rendered_ignore = dependency_owned_public_names(DataDrivenDMD),
    ),
    # `get_trainerror`/`get_testerror` are listed in `export` (src/DataDrivenDMD.jl) but
    # no method is ever defined for them (dead exports from the #371 refactor). Aqua was
    # never run before this conversion, so this is pre-existing and surfaced, not
    # introduced; marked broken rather than silenced.
    aqua_broken = (:undefined_exports,),
    ei_kwargs = (;
        # DataDrivenDiffEq exposes these names as the (non-`export`ed) extension API its
        # own sublibraries build on. They are used deliberately; declaring them `public`
        # upstream is tracked separately.
        all_explicit_imports_are_public = (;
            ignore = (
                :AbstractBasis, :AbstractDataDrivenAlgorithm, :AbstractDataDrivenResult,
                :ABSTRACT_CONT_PROB, :ABSTRACT_DISCRETE_PROB,
                :InternalDataDrivenProblem, :is_controlled, :is_implicit,
            ),
        ),
        all_qualified_accesses_are_public = (;
            ignore = (:__EMPTY_MATRIX, :__construct_basis, :get_fit_targets),
        ),
    ),
    # The umbrella `using DataDrivenDiffEq` (plus the `using DataDrivenDiffEq.<submodule>`
    # re-exports) pulls the DataDrivenDiffEq public surface in implicitly; making every
    # name explicit is a large refactor tracked separately.
    ei_broken = (:no_implicit_imports,)
)
