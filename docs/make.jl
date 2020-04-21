using Documenter, DataDrivenDiffEq

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
        "Equation Free Systems" => Any[
            "koopman/dmd.md",
        ],
        "Equation Driven Systems" => Any[
            "sparse_identification/sindy.md",
        ]
        #"highlevel.md",
        #"Systems" => Any[
        #    "systems/AbstractSystem.md",
        #    "systems/ODESystem.md",
        #    "systems/SDESystem.md",
        #    "systems/NonlinearSystem.md",
        #    "systems/OptimizationSystem.md",
        #    "systems/ReactionSystem.md",
        #    "systems/PDESystem.md"
        #],
        #"IR.md"
    ]
)

deploydocs(
   repo = "github.com/SciML/DataDrivenDiffEq.jl.git";
   push_preview = true
)
