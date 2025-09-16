using Documenter
using DocumenterInterLinks
using Revise

using PointCloudRegistration
using CoordinateTransformations

Revise.revise()

links = InterLinks(
    "CoordinateTransformations" => (
        "https://juliageometry.github.io/CoordinateTransformations.jl/dev/",
        joinpath(@__DIR__, "inventories", "coordinatetransformations.inv"),
    ),
)

makedocs(;
    sitename = "PointCloudRegistration.jl",
    remotes = nothing,
    plugins = [links],
    format = Documenter.HTMLWriter.HTML(;
        canonical = "a5s.eu/PointCloudRegistration.jl/",
        repolink = "https://codeberg.org/andreas-k/PointCloudRegistration.jl",
        assets = [asset("assets/logo.png"; islocal = true, class = :ico)],
    ),
)
