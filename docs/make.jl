using Pkg
Pkg.activate("..")

using Documenter, DataDrivenDiffEq
#include("../src/DataDrivenDiffEq.jl")

makedocs(
    sitename="DataDrivenDiffEq.jl",
    authors="Julius Martensen, Chris Rackauckas",
    modules=[DataDrivenDiffEq],
    clean=true,doctest=false,
    format = Documenter.HTML(#analytics = "UA-90474609-3",
                             assets = ["assets/favicon.ico"],
                             canonical="https://docs.sciml.ai/stable/"),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "quickstart.md",
        "Basis" => "basis.md",
        "Koopman Operators" => Any[
            "koopman/koopman.md",
            "koopman/dmd.md",
            "koopman/edmd.md",
            "koopman/dmdc.md",
            "koopman/algorithms.md"
        ],
        "Sparse Identification" => Any[
            "sparse_identification/sindy.md"
        ]
        #"Equation Free Systems" => Any[
        #    "koopman/dmd.md",
        #],
        #"Equation Driven Systems" => Any[
        #    "sparse_identification/sindy.md",
        #]
    ]
)

deploydocs(
   repo = "github.com/SciML/DataDrivenDiffEq.jl.git";
   push_preview = true
)
