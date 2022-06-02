#push!(LOAD_PATH,"../src/")

using Documenter, DataDrivenDiffEq
using Flux, SymbolicRegression
using Literate

include("pages.jl")

ENV["GKSwstype"] = "100"

# Evaluate the example directory

src = joinpath(@__DIR__, "src")
lit = joinpath(@__DIR__, "examples")
excludes = []#["symbolic_regression.jl"]
tutorials = []

for (root, _, files) ∈ walkdir(lit), file ∈ files
  file ∈ excludes && continue
  fname, fext = splitext(file)

  fext == ".jl" || continue
  ipath = joinpath(root, file)
  opath = joinpath(splitdir(replace(ipath, lit=>src))[1], "examples")
  script = Literate.script(ipath, opath, execute = false, comments = false)
  code = strip(read(script, String))
  mdpost(str) = replace(str, "@__CODE__" => code)
  Literate.markdown(ipath, opath)
  Literate.markdown(ipath, opath, execute = false, postprocess = mdpost)
  if fname == "0_getting_started"
    pushfirst!(tutorials,  relpath(joinpath(opath, fname*".md"), src))
  else
    push!(tutorials, relpath(joinpath(opath, fname*".md"), src))
  end
end

# Create the docs
makedocs(
    sitename="DataDrivenDiffEq.jl",
    authors="Julius Martensen, Christopher Rackauckas",
    modules=[DataDrivenDiffEq],
    clean=true,doctest=false,
    format = Documenter.HTML(analytics = "UA-90474609-3",
                             assets = ["assets/favicon.ico"],
                             canonical="https://datadriven.sciml.ai/stable/"),
    pages=pages
)

deploydocs(
   repo = "github.com/SciML/DataDrivenDiffEq.jl.git";
   push_preview = true
)
