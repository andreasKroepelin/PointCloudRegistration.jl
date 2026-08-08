module EmdbHelper

using MRCFile
using DimensionalData
using Downloads

function load_map(id)
    filename = "emd_$id.map.gz"
    if !isfile(filename)
        url = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/" *
            "EMD-$id/map/emd_$id.map.gz"
        @info "downloading from EMDB..." url filename
        Downloads.download(url, filename)
    end
    read(filename, MRCData)
end

function mrc2dimarr(mrc)
    axs = voxelaxes(header(mrc))
    dimaxs = map((D, ax) -> D(ax), (X, Y, Z), axs)
    DimArray(mrc.data, dimaxs)
end

function animate_plot_rotation(fig, ax; framerate = 30)
    Record(fig, range(0, 2pi; length = 100); framerate) do azimuth
        ax.azimuth[] = azimuth
    end
end

end # module EmdbHelper
