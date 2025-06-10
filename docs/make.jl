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
)
