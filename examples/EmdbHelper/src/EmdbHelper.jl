module EmdbHelper

using MRCFile
using DimensionalData
using Downloads

function load_map(id)
    filename = "emd_$id.map.gz"
    if !isfile(filename)
        Downloads.download(
            "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/" *
                "EMD-$id/map/emd_$id.map.gz",
            filename
        )
    end
    read(filename, MRCData)
end

function mrc2dimarr(mrc)
    axs = voxelaxes(header(mrc))
    dimaxs = map((D, ax) -> D(ax), (X, Y, Z), axs)
    DimArray(mrc.data, dimaxs)
end

end # module EmdbHelper
