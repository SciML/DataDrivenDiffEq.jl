using Documenter, DataDrivenDiffEq

ENV["GKSwstype"] = "100"

makedocs(
    sitename="DataDrivenDiffEq.jl",
    authors="Julius Martensen, Chris Rackauckas",
    modules=[DataDrivenDiffEq],
    clean=true,doctest=false,
    format = Documenter.HTML(#analytics = "UA-90474609-3",
                             assets = ["assets/favicon.ico"],
                             canonical="https://datadriven.sciml.ai/stable/"),
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
            "sparse_identification/sindy.md",
            "sparse_identification/isindy.md",
            "sparse_identification/optimizers.md"
        ],
        "Utilities" => "utils.md",
        "Contributing" => "contributions.md"
    ]
)

deploydocs(
   repo = "github.com/SciML/DataDrivenDiffEq.jl.git";
   push_preview = true
)
