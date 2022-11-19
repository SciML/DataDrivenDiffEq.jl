using SafeTestsets, Pkg

@info "Finished loading packages"

const GROUP = get(ENV, "GROUP", "All")

function dev_subpkg(subpkg)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", subpkg)
    Pkg.develop(PackageSpec(path = subpkg_path))
end

function activate_subpkg_env(subpkg)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", subpkg)
    Pkg.activate(subpkg_path)
    Pkg.develop(PackageSpec(path = subpkg_path))
    Pkg.instantiate()
end

@time begin if GROUP == "All" || GROUP == "Core" || GROUP == "Downstream"
    @safetestset "Basis" begin include("./basis/basis.jl") end
    @safetestset "Implicit Basis" begin include("./basis/implicit_basis.jl") end
    @safetestset "Basis generators" begin include("./basis/generators.jl") end
    @safetestset "DataDrivenProblem" begin include("./problem/problem.jl") end
    @safetestset "DataDrivenSolution" begin include("./solution/solution.jl") end
    @safetestset "Utilities" begin include("./utils.jl") end
    @safetestset "CommonSolve" begin include("./commonsolve/commonsolve.jl") end
else
    dev_subpkg(GROUP)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", GROUP)
    Pkg.test(PackageSpec(name = GROUP, path = subpkg_path); coverage = true)
end end
