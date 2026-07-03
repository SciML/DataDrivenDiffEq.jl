using SciMLTesting
using DataDrivenDiffEq
using Aqua
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
    # `deleteat!`/`unique!` are defined on `Symbolics.Num`/`Symbolics.Arr` vectors in
    # src/basis/type.jl as the public Basis-manipulation API. Aqua flags them as type
    # piracy and, as a direct consequence, reports the resulting method ambiguities
    # against `Base.unique!`/`Base.deleteat!`. `DataDrivenDataset` uses `Vararg{T, N}`
    # where `N` is unbound for the zero-argument signature. All three are pre-existing
    # design properties (Aqua was never run before this conversion), surfaced—not
    # introduced—here, so they are marked broken rather than silenced. `aqua_broken`
    # also disables each named sub-check in the `Aqua.test_all` call.
    aqua_broken = (:ambiguities, :unbound_args, :piracies),
    ei_kwargs = (;
        # All flagged names are non-public (un-`export`ed / not `public`-declared)
        # internals of upstream packages, accessed deliberately. They are owned by the
        # listed modules; making them public is an upstream change tracked separately.
        all_qualified_accesses_are_public = (;
            ignore = (
                :NullParameters,          # DiffEqBase
                :Tunable, :canonicalize,  # SciMLStructures
                :isscimlstructure,        # SciMLStructures
                :derivative,              # DataInterpolations
                :getdefaultval, :setdefaultval, :symtype,  # Symbolics
                :isconst, :promote_symtype,                # SymbolicUtils
                :nameof, :toparam, :tovar,                 # ModelingToolkit / Base.nameof method
                :promote_eltype,          # Base
                :transform!,              # StatsBase
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                :Operator,  # Symbolics (owner SymbolicUtils): `Difference <: Operator`
                :issym,     # SymbolicUtils
            ),
        ),
        # `Operator` is re-exported by Symbolics from SymbolicUtils; importing it via the
        # `Symbolics` namespace is intentional (Symbolics is the user-facing surface here).
        all_explicit_imports_via_owners = (;
            ignore = (:Operator,),
        ),
        # `NullParameters`/`nameof`/`symtype`/`toparam`/`tovar` are accessed through a
        # re-exporting/owning namespace rather than `Base.which`'s reported owner.
        all_qualified_accesses_via_owners = (;
            ignore = (:NullParameters, :nameof, :symtype, :toparam, :tovar),
        ),
    ),
    # The `@reexport using ModelingToolkit/StatsBase/DataInterpolations/MLUtils/CommonSolve`
    # surface is pulled in implicitly by design (this package re-exports it); making every
    # name explicit is a large refactor tracked separately.
    ei_broken = (:no_implicit_imports,)
)
