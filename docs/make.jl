push!(LOAD_PATH,"../src/")

using Documenter, DataDrivenDiffEq
using Flux, SymbolicRegression
using Literate

ENV["GKSwstype"] = "100"

# Evaluate the example directory
# This workflow is based on the implementation of JuAFem.jl

example_files = [
  "linear_discrete_system.jl", "linear_continuous_system.jl"
]
example_dir = joinpath(@__DIR__, "examples")
output_dir = joinpath(@__DIR__, "src", "examples")

for example in example_files
  s_ = joinpath(example_dir, example)
  script = Literate.script(s_, output_dir, execute = false, comments = false)
  code = strip(read(script, String))
  mdpost(str) = replace(str, "@__CODE__" => code)
  Literate.markdown(s_, output_dir, execute = false, postprocess = mdpost)
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
        "Tutorials" => [joinpath("examples", split(f, ".")[1]*".md") for f in example_files],
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
