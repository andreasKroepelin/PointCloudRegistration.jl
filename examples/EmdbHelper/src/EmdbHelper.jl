module EmdbHelper

using MRCFile

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

end # module EmdbHelper
