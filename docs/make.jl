using Documenter
using BendersX

makedocs(
    modules = [BendersX],
    sitename = "BendersX.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://asu-opt-lab.github.io/BendersX.jl/",
    ),
    checkdocs = :public,
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Getting Started" => "tutorials/getting_started.md",
            "Swapping Oracles and Adjusting Their Behaviors" => "tutorials/oracles.md",
            "Swapping Environments and Adjusting Their Behaviors" => "tutorials/envs.md",
            "Examples" => "tutorials/examples.md",
        ],
        "User Guide" => [
            "Core Architecture" => "user_guide.md",
            "MPI and Distributed Execution" => "mpi.md",
        ],
        "API" => "api.md",
    ],
    # sitelogo = "assets/logo.svg",
)

deploydocs(
    repo = "github.com/asu-opt-lab/BendersX.jl.git",
    devbranch = "main",
    push_preview = true,
    versions = nothing,
)
