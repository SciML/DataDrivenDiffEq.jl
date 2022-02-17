#push!(LOAD_PATH,"../src/")

using Documenter, DataDrivenDiffEq
#using Flux, SymbolicRegression
using Literate

ENV["GKSwstype"] = "100"

# Evaluate the example directory

src = joinpath(@__DIR__, "src")
lit = joinpath(@__DIR__, "examples")

tutorials = []

for (root, _, files) ∈ walkdir(lit), file ∈ files
  fname, fext = splitext(file)
  fext == ".jl" || continue
  ipath = joinpath(root, file)
  opath = joinpath(splitdir(replace(ipath, lit=>src))[1], "examples")
  script = Literate.script(ipath, opath, execute = false, comments = false)
  code = strip(read(script, String))
  mdpost(str) = replace(str, "@__CODE__" => code)
  Literate.markdown(ipath, opath)
  Literate.markdown(ipath, opath, execute = false, postprocess = mdpost)
  push!(tutorials, relpath(joinpath(opath, fname*".md"), src))
end


# Create the docs
makedocs(
    sitename="DataDrivenDiffEq.jl",
    authors="Julius Martensen, Christopher Rackauckas",
    modules=[DataDrivenDiffEq],
    clean=true,doctest=false,
    format = Documenter.HTML(#analytics = "UA-90474609-3",
                             assets = ["assets/favicon.ico"],
                             canonical="https://datadriven.sciml.ai/stable/"),
    pages=[
        "Home" => "index.md",
        #"Getting Started" => "getting_started.md",
        "Tutorials" => tutorials,
        #"Unifying SINDy and DMD" => "sindy_dmd.md",
        #"Problems" => "problems.md",
        #"Basis" => "basis.md",
        #"Solvers" => Any[
        #  "solvers/koopman.md",
        #  "solvers/optimization.md",
        #  "solvers/symbolic_regression.md"
        #],
        #"Solutions" => "solutions.md",
        #"Utilities" => "utils.md",
        #"Contributing" => "contributions.md",
        #"Citing" => "citations.md"
        ]
)

deploydocs(
   repo = "github.com/SciML/DataDrivenDiffEq.jl.git";
   push_preview = true
)
