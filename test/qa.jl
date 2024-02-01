using DataDrivenDiffEq, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(DataDrivenDiffEq)
    Aqua.test_ambiguities(DataDrivenDiffEq, recursive = false, broken = true)
    Aqua.test_deps_compat(DataDrivenDiffEq)
    Aqua.test_piracies(DataDrivenDiffEq, broken = true)
    Aqua.test_project_extras(DataDrivenDiffEq)
    Aqua.test_stale_deps(DataDrivenDiffEq)
    Aqua.test_unbound_args(DataDrivenDiffEq, broken = true)
    Aqua.test_undefined_exports(DataDrivenDiffEq, broken = true)
end
