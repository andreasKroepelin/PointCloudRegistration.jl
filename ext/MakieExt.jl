module MakieExt
using PointCloudRegistration
using PointCloudRegistration: avg_nn_dist, max_of_weights
# this makes the Makie recipe add methods to the functions defined in the main
# package, so we can access them
import PointCloudRegistration:
    pointcloudplotflat,
    pointcloudplotmesh,
    scaffoldplotflat,
    scaffoldplotmesh,
    pointcloudplotflat!,
    pointcloudplotmesh!,
    scaffoldplotflat!,
    scaffoldplotmesh!
using Makie
using Makie.Markdown
using Makie.GeometryBasics
using Distances

function _ext_makie_docref(olddoc::Markdown.MD)
    olddoc_code = Markdown.plain(olddoc)
    newdoc_code =
        replace(olddoc_code, r"(\[`Makie\..+`\])\(@ref\)" => s"\1(@extref)")
    return Markdown.parse(newdoc_code)
end

autosizefactor(pc) = avg_nn_dist(pc) / max_of_weights(pc)

maybe_collect(xs::AbstractVector) = collect(xs)
maybe_collect(xs::Vector) = xs

const PointCloud2Or3 = Union{<:PointCloud{2}, <:PointCloud{3}}

"""
    pointcloudplotflat(pointcloud)

Plots the point cloud using a regular scatter plot of its points.
The size of each marker is proportional to the point's weight.
You can control this relationship via the `sizefactor` attribute.
"""
@recipe PointCloudPlotFlat (pointcloud::PointCloud2Or3,) begin
    Makie.documented_attributes(Scatter)...
    """
    Defines the ratio between marker size and weight of the corresponding point.
    That is, a point with weight `w` will have a marker size of
    `sizefactor * w`.
    Defaults to the average nearest neighbor distance in the point cloud
    divided by the maximum weight.
    """
    sizefactor = Makie.automatic
end

function Makie.plot!(plot::PointCloudPlotFlat)
    map!(plot.attributes, [:pointcloud, :sizefactor], [:positions, :sizes]) do pc, sizefactor
        sf = sizefactor isa Makie.Automatic ? autosizefactor(pc) : sizefactor
        ps = sf .* weights(pc)
        (points(pc), maybe_collect(ps))
    end
    scatter!(
        plot,
        plot.attributes,
        plot.positions;
        markersize = plot.sizes,
        markerspace = :data,
    )
end

"""
    pointcloudplotmesh(pointcloud)

Plots the point cloud using a mesh scatter plot of its points.
The size of each marker is proportional to the point's weight.
You can control this relationship via the `sizefactor` attribute.
"""
@recipe PointCloudPlotMesh (pointcloud::PointCloud{3},) begin
    Makie.documented_attributes(MeshScatter)...
    """
    Defines the ratio between marker size and weight of the corresponding point.
    That is, a point with weight `w` will have a marker size of
    `sizefactor * w`.
    Defaults to the average nearest neighbor distance in the point cloud
    divided by the maximum weight.
    """
    sizefactor = Makie.automatic
end

function Makie.plot!(plot::PointCloudPlotMesh)
    map!(plot.attributes, [:pointcloud, :sizefactor], [:positions, :sizes]) do pc, sizefactor
        sf = sizefactor isa Makie.Automatic ? autosizefactor(pc) : sizefactor
        ps = sf .* weights(pc)
        (points(pc), maybe_collect(ps))
    end
    meshscatter!(plot, plot.attributes, plot.positions; markersize = plot.sizes)
end

Makie.plottype(::PointCloud{2}) = PointCloudPlotFlat
Makie.plottype(::PointCloud{3}) = PointCloudPlotMesh
Makie.preferred_axis_type(::PointCloudPlotMesh) = Makie.LScene

Makie.convert_arguments(
    al::Makie.ArrowLike,
    displacement::PointCloudRegistration.Displacement,
) = Makie.convert_arguments(
    al,
    displacement.origin,
    displacement.result .- displacement.origin,
)

function _extract_segments(
    prep::PointCloudRegistration.PreparedSourceDistPres,
    pointcloud::PointCloud2Or3,
)
    pts = points(pointcloud)
    n = length(prep.neighbor_edges)
    endpoints = fill((zero(eltype(pts)), zero(eltype(pts))), n)
    zero_dist = euclidean(pts[1], pts[1])
    one_dist = oneunit(zero_dist)
    stretches = fill(log(one_dist / one_dist), n)
    maxd = typemin(zero_dist)
    for idx in Base.OneTo(n)
        ne = prep.neighbor_edges[idx]
        (i, j) = ne.from_to
        x = pts[i]
        y = pts[j]
        d = euclidean(x, y)
        maxd = max(d, maxd)
        s = abs(log(d / ne.distance))
        endpoints[idx] = (x, y)
        stretches[idx] = s
    end
    return (; endpoints, stretches, autoradius = maxd / 10)
end

"""
    scaffoldplotflat(prepared_source, pointcloud)

Plots the scaffold that is aimed to be preserved during `DistancePreserving`
non-rigid registration using line segments.
The `prepared_source` for the first argument can be obtained via
[`prepare_source_distancepreserving`](@ref).

Note that it is not required that `pointcloud` was used for
`prepare_source_distancepreserving`.
It is only relevant that it has the same "meaning" of the points.
This especially means that you can use the non-rigidly registered version of
that point cloud here as well.
"""
@recipe ScaffoldPlotFlat (
    prep::PointCloudRegistration.PreparedSourceDistPres,
    pointcloud::PointCloud2Or3,
) begin
    Makie.filtered_attributes(LineSegments; exclude = [])...
    """
    If set to `true` (default), lines are colored according to the
    logarithmic relative stretch (that is,
    `abs(log(distance / original_distance))`).
    Otherwise, they all have the given `color`.
    """
    color_by_stretch = true
end

@doc (_ext_makie_docref(@doc scaffoldplotflat)) scaffoldplotflat

function Makie.plot!(plot::ScaffoldPlotFlat)
    map!(
        plot.attributes,
        [:prep, :pointcloud, :color, :color_by_stretch],
        [:endpoints, :stretch_color]
    ) do prep, pc, color, color_by_stretch
        (; endpoints, stretches) = _extract_segments(prep, pc)
        stretch_color = color_by_stretch ? stretches : color
        return (endpoints, stretch_color)
    end
    linesegments!(plot, plot.attributes, plot.endpoints; color = plot.stretch_color)
end

Makie.plottype(
    ::PointCloudRegistration.PreparedSourceDistPres,
    ::PointCloud{2},
) = ScaffoldPlotFlat

"""
    scaffoldplotmesh(prepared_source, pointcloud)

Plots the scaffold that is aimed to be preserved during `DistancePreserving`
non-rigid registration using cylinder meshes.
The `prepared_source` for the first argument can be obtained via
[`prepare_source_distancepreserving`](@ref).

Note that it is not required that `pointcloud` was used for
`prepare_source_distancepreserving`.
It is only relevant that it has the same "meaning" of the points.
This especially means that you can use the non-rigidly registered version of
that point cloud here as well.
"""
@recipe ScaffoldPlotMesh (
    prep::PointCloudRegistration.PreparedSourceDistPres,
    pointcloud::PointCloud{3},
) begin
    Makie.filtered_attributes(Makie.Mesh; exclude = [])...
    """
    If set to `true` (default), cylinders are colored according to the
    logarithmic relative stretch (that is,
    `abs(log(distance / original_distance))`).
    Otherwise, they all have the given `color`.
    """
    color_by_stretch = true
    """
    The radius of each cylinder.
    """
    radius = Makie.automatic
end

@doc (_ext_makie_docref(@doc scaffoldplotmesh)) scaffoldplotmesh

function Makie.plot!(plot::ScaffoldPlotMesh)
    map!(plot.attributes, [:prep, :pointcloud, :radius, :color, :color_by_stretch], [:combined_mesh, :stretch_color]) do prep, pc, radius, color, color_by_stretch
        (; endpoints, stretches, autoradius) = _extract_segments(prep, pc)
        r = Makie.default_automatic(radius, autoradius)
        meshes = [
            triangle_mesh(Cylinder(Point3(x), Point3(y), r))
            for (x, y) in endpoints
        ]
        combined_mesh = merge(meshes)
        stretch_color = if color_by_stretch
            nvertices = length(meshes[1].position) # should be 32 but let's be sure
            @assert all(m -> length(m.position) == nvertices, meshes)
            # Makie.mesh() wants one color per mesh vertex
            repeat(stretches; inner = nvertices)
        else
            color
        end
        return (combined_mesh, stretch_color)
    end
    mesh!(plot, plot.attributes, plot.combined_mesh; color = plot.stretch_color)
end

Makie.plottype(
    ::PointCloudRegistration.PreparedSourceDistPres,
    ::PointCloud{3},
) = ScaffoldPlotMesh
Makie.preferred_axis_type(::ScaffoldPlotMesh) = Makie.LScene

end
