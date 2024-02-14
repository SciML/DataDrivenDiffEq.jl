#push!(LOAD_PATH,"../src/")

# Add the sublibs
using Pkg

function dev_subpkg(subpkg)
    subpkg_path = abspath(joinpath(dirname(@__FILE__), "..", "lib", subpkg))
    Pkg.develop(PackageSpec(path = subpkg_path))
end

dev_subpkg("DataDrivenDMD")
dev_subpkg("DataDrivenSparse")
dev_subpkg("DataDrivenSR")

using Documenter
using DataDrivenDiffEq
using DataDrivenDMD
using DataDrivenSparse
using DataDrivenSR

using StatsBase
using Literate

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

ENV["GKSwstype"] = "100"

# Evaluate the example directory
src = joinpath(@__DIR__, "src")

function create_tutorials(dirname, targetdir, excludes = [])
    tutorials = []

    if isdir(targetdir)
        rm(targetdir, recursive = true)
    else
        mkdir(targetdir)
    end

    foreach(walkdir(dirname)) do (root, _, files)
        for file in files
            file ∈ excludes && continue
            fname, fext = splitext(file)
            fext == ".jl" || continue
            ipath = joinpath(root, file)
            script = Literate.script(ipath, targetdir, execute = false, comments = false)
            @info script
            code = strip(read(script, String))
            mdpost(str) = replace(str, "@__CODE__" => code)
            Literate.markdown(ipath, targetdir)
            Literate.markdown(ipath, targetdir, execute = false, postprocess = mdpost)
            push!(tutorials,
                relpath(joinpath(targetdir, fname * ".md"), joinpath(@__DIR__, "src")))
        end
    end
    return tutorials
end

koopman_tutorial = create_tutorials(joinpath(@__DIR__, "src/libs/datadrivendmd/"),
    joinpath(@__DIR__, "src/libs/datadrivendmd/examples"))
sparse_tutorial = create_tutorials(joinpath(@__DIR__, "src/libs/datadrivensparse/"),
    joinpath(@__DIR__, "src/libs/datadrivensparse/examples"))
sr_tutorial = create_tutorials(joinpath(@__DIR__, "src/libs/datadrivensr/"),
    joinpath(@__DIR__, "src/libs/datadrivensr/examples"))

# Must be after tutorials is created
include("pages.jl")

# Create the docs
makedocs(sitename = "DataDrivenDiffEq.jl",
    authors = "Julius Martensen, Christopher Rackauckas, et al.",
    modules = [DataDrivenDiffEq, DataDrivenDMD, DataDrivenSparse, DataDrivenSR],
    clean = true, doctest = false, linkcheck = true,
    warnonly = [:missing_docs, :cross_references],
    linkcheck_ignore = ["http://cwrowley.princeton.edu/papers/Hemati-2017a.pdf",
        "https://royalsocietypublishing.org/doi/10.1098/rspa.2020.0279",
        "https://www.pnas.org/doi/10.1073/pnas.1517384113"],
    format = Documenter.HTML(assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/DataDrivenDiffEq/stable/"),
    pages = pages)

deploydocs(repo = "github.com/SciML/DataDrivenDiffEq.jl.git";
    push_preview = true)
