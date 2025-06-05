using Documenter
using DocumenterInterLinks

using PointCloudRegistration
using CoordinateTransformations

links = InterLinks(
    "CoordinateTransformations" => "https://juliageometry.github.io/CoordinateTransformations.jl/dev/",
)

makedocs(sitename="PointCloudRegistration.jl", remotes = nothing, plugins = [links])
