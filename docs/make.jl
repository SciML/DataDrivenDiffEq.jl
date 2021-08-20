push!(LOAD_PATH,"../src/")

using Documenter, DataDrivenDiffEq
using Flux, SymbolicRegression

ENV["GKSwstype"] = "100"

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
        "Getting Started" => "getting_started.md",
        "Tutorials" => Any[
          "examples/linear_systems.md",
          "examples/nonlinear_systems.md",
          "examples/real_world.md"
        ],
        "Unifying SINDy and DMD" => "sindy_dmd.md",
        "Problems" => "problems.md",
        "Basis" => "basis.md",
        "Solvers" => Any[
          "solvers/koopman.md",
          "solvers/optimization.md",
          "solvers/symbolic_regression.md"
        ],
        "Solutions" => "solutions.md",
        "Utilities" => "utils.md",
        "Contributing" => "contributions.md",
        "Citing" => "citations.md"
        ]
)

deploydocs(
   repo = "github.com/SciML/DataDrivenDiffEq.jl.git";
   push_preview = true
)
