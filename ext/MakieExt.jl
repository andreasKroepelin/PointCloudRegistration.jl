module MakieExt
using PointCloudRegistration
using PointCloudRegistration: avg_nn_dist, max_of_weights
# this makes the Makie recipe add methods to the functions defined in the main
# package, so we can access them
import PointCloudRegistration:
    pointcloudplotflat,
    pointcloudplotmesh,
    scaffoldplot,
    pointcloudplotflat!,
    pointcloudplotmesh!,
    scaffoldplot!
using Makie
using Makie.Markdown

function _ext_makie_docref(olddoc::Markdown.MD)
    olddoc_code = Markdown.plain(olddoc)
    newdoc_code =
        replace(olddoc_code, r"(\[`Makie\..+`\])\(@ref\)" => s"\1(@extref)")
    return Markdown.parse(newdoc_code)
end

function _plotsizes(
    pc::PointCloud,
    sizefactor = avg_nn_dist(pc) / max_of_weights(pc),
)
    sizefactor .* weights(pc)
end

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
    sizefactor = :auto
end

function Makie.plot!(plot::PointCloudPlotFlat)
    map!(plot.attributes, [:pointcloud], [:positions, :sizes]) do pc
        sf = plot.sizefactor[]
        ps = if sf == :auto
            _plotsizes(pc)
        elseif sf isa Number
            _plotsizes(pc, sf)
        end
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
    sizefactor = :auto
end

function Makie.plot!(plot::PointCloudPlotMesh)
    map!(plot.attributes, [:pointcloud], [:positions, :sizes]) do pc
        sf = plot.sizefactor[]
        ps = if sf == :auto
            _plotsizes(pc)
        elseif sf isa Number
            _plotsizes(pc, sf)
        end
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

"""
    scaffoldplot(prepared_source, pointcloud)

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
@recipe ScaffoldPlot (
    prep::PointCloudRegistration.PreparedSourceDistPres,
    pointcloud::PointCloud2Or3,
) begin
    Makie.documented_attributes(LineSegments)...
end

@doc (_ext_makie_docref(@doc scaffoldplot)) scaffoldplot

function Makie.plot!(plot::ScaffoldPlot)
    map!(plot.attributes, [:prep, :pointcloud], :segments) do prep, pc
        [(pc[i].coords, pc[j].coords) for (i, j) in prep.neighbor_edges.from_to]
    end
    linesegments!(plot, plot.attributes, plot.segments;)
end

Makie.plottype(
    ::PointCloudRegistration.PreparedSourceDistPres,
    ::PointCloud2Or3,
) = ScaffoldPlot

end
