using Documenter
using DocumenterInterLinks
using Revise
using Literate

using PointCloudRegistration
using CoordinateTransformations

Revise.revise()

links = InterLinks(
    "CoordinateTransformations" => (
        "https://juliageometry.github.io/CoordinateTransformations.jl/dev/",
        joinpath(@__DIR__, "inventories", "coordinatetransformations.inv"),
    ),
)

# the following is from
# https://discourse.julialang.org/t/multiple-environments-in-documenter/63436/4
EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
OUTPUT_DIR = joinpath(@__DIR__, "src", "generated")
examples = [
    "ribosome-puzzle/ribosome-puzzle.jl",
    # "wglmakie-test/wglmakie-test.jl",
]
jlcmd = Base.julia_cmd()
for example in examples
    example_path = joinpath(EXAMPLES_DIR, example)
    load_path = "$(dirname(example_path)):$(@__DIR__)"
    code = """
    using Literate
    Literate.markdown("$example_path", "$OUTPUT_DIR"; execute = true)
    """
    run(addenv(`$(jlcmd) -e $(code)`, Dict("JULIA_LOAD_PATH" => load_path)))
end

makedocs(;
    sitename = "PointCloudRegistration.jl",
    pages = [
        "index.md",
        "Examples" => [
            "generated/ribosome-puzzle.md",
            "generated/wglmakie-test.md",
        ],
    ],
    remotes = nothing,
    plugins = [links],
    format = Documenter.HTMLWriter.HTML(;
        canonical = "a5s.eu/PointCloudRegistration.jl/",
        repolink = "https://codeberg.org/andreas-k/PointCloudRegistration.jl",
        assets = [asset("assets/logo.png"; islocal = true, class = :ico)],
        size_threshold_ignore = [
            "generated/ribosome-puzzle.md",
            "generated/wglmakie-test.md",
        ],
    ),
)
