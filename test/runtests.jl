using SafeTestsets, Test, Pkg
using SciMLTesting

@info "Finished loading packages"

const GROUP = get(ENV, "GROUP", "All")

# GROUP can name a sublibrary (e.g. "DataDrivenDMD" -> Core test group) or
# "{sublibrary}_{TEST_GROUP}" for a custom group (e.g. "DataDrivenDMD_QA").
# Sublibraries declare any non-default groups in test/test_groups.toml; the
# DATADRIVENDIFFEQ_TEST_GROUP env var threads the group into the sublib's
# own runtests.jl. CI dispatches sublibraries through SublibraryCI.yml, so
# this branch is primarily for running a sublibrary's suite locally.
@time begin
    lib_dir = joinpath(dirname(@__DIR__), "lib")
    base_group, test_group = detect_sublibrary_group(GROUP, lib_dir)

    if !isempty(base_group) && isdir(joinpath(lib_dir, base_group))
        Pkg.activate(joinpath(lib_dir, base_group))
        # On Julia < 1.11 the [sources] section is not honored; develop the
        # in-repo path dependencies (transitively) so the sublibrary tests run
        # against this checkout of DataDrivenDiffEq rather than a released one.
        if VERSION < v"1.11.0-DEV.0"
            developed = Set{String}()
            push!(developed, normpath(joinpath(lib_dir, base_group)))
            specs = Pkg.PackageSpec[]
            queue = [joinpath(lib_dir, base_group)]
            while !isempty(queue)
                pkg_dir = popfirst!(queue)
                toml_path = joinpath(pkg_dir, "Project.toml")
                isfile(toml_path) || continue
                toml = Pkg.TOML.parsefile(toml_path)
                if haskey(toml, "sources")
                    for (dep_name, source_spec) in toml["sources"]
                        if source_spec isa Dict && haskey(source_spec, "path")
                            dep_path = normpath(joinpath(pkg_dir, source_spec["path"]))
                            if isdir(dep_path) && !(dep_path in developed)
                                push!(developed, dep_path)
                                @info "Queuing local source dependency" dep_name dep_path
                                push!(specs, Pkg.PackageSpec(path = dep_path))
                                push!(queue, dep_path)
                            end
                        end
                    end
                end
            end
            isempty(specs) || Pkg.develop(specs)
        end
        withenv("DATADRIVENDIFFEQ_TEST_GROUP" => test_group) do
            Pkg.test(base_group, coverage = true)
        end
    else
        # QA is a dep-adding group (JET in test/qa/Project.toml), excluded from
        # `All` (curated to Core) and not run on prerelease Julia (enforced by
        # test/test_groups.toml versions).
        run_tests(;
            core = function ()
                @safetestset "Basis" begin
                    include("./Core/basis.jl")
                end
                @safetestset "Implicit Basis" begin
                    include("./Core/implicit_basis.jl")
                end
                @safetestset "Basis generators" begin
                    include("./Core/generators.jl")
                end
                @safetestset "DataDrivenProblem" begin
                    include("./Core/problem.jl")
                end
                @safetestset "DataDrivenSolution" begin
                    include("./Core/solution.jl")
                end
                @safetestset "Utilities" begin
                    include("./Core/utils.jl")
                end
                @safetestset "CommonSolve" begin
                    include("./Core/commonsolve.jl")
                end
            end,
            qa = (;
                env = joinpath(@__DIR__, "qa"),
                body = function ()
                    @safetestset "Quality Assurance" begin
                        include("qa/qa.jl")
                    end
                    @safetestset "JET Static Analysis" begin
                        include("qa/jet_tests.jl")
                    end
                end,
            ),
            all = ["Core"],
        )
    end
end
