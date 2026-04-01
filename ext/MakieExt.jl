module MakieExt
using PointCloudRegistration
using PointCloudRegistration: avg_nn_dist
using Makie

function _plotsizes(
    pc::PointCloud,
    sizefactor = avg_nn_dist(pc) / maximum(pc.weights)
)
    sizefactor .* pc.weights
end

maybe_collect(xs::AbstractVector) = collect(xs)
maybe_collect(xs::Vector) = xs

const PointCloud2Or3 = Union{<:PointCloud{2}, <:PointCloud{3}}

@recipe PointCloudPlotFlat (pointcloud::PointCloud2Or3,) begin
    Makie.documented_attributes(Scatter)...
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
        (pc.points, maybe_collect(ps))
    end
    scatter!(
        plot,
        plot.attributes,
        plot.positions;
        markersize = plot.sizes,
        markerspace = :data,
    )
end

@recipe PointCloudPlotMesh (pointcloud::PointCloud{3},) begin
    Makie.documented_attributes(MeshScatter)...
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
        (pc.points, maybe_collect(ps))
    end
    meshscatter!(
        plot,
        plot.attributes,
        plot.positions;
        markersize = plot.sizes,
    )
end

Makie.plottype(::PointCloud{2}) = PointCloudPlotFlat
Makie.plottype(::PointCloud{3}) = PointCloudPlotMesh
Makie.preferred_axis_type(::PointCloudPlotMesh) = Makie.LScene

end
