module MakieExt
using PointCloudRegistration
using PointCloudRegistration: avg_nn_dist
using Makie

@recipe PointCloudPlot (pointcloud::PointCloud,) begin
    color = @inherit markercolor
    Makie.mixin_generic_plot_attributes()...
end

function Makie.plot!(plot::PointCloudPlot)
    map!(plot.attributes, [:pointcloud], [:positions, :sizes]) do pc
        sizefactor = avg_nn_dist(pc) / maximum(pc.weights)
        (pc.points, sizefactor .* pc.weights)
    end
    scatter!(
        plot,
        plot.positions;
        markersize = plot.sizes,
        markerspace = :data, plot.color
    )
end

Makie.plottype(::PointCloud) = PointCloudPlot

end
