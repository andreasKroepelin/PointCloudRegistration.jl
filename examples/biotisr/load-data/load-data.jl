using PythonCall
using CondaPkg
using Downloads
using ZipArchives: ZipReader, zip_names, zip_readentry
using JLD2

function @main(args)
    CondaPkg.add("mrcfile")
    mrcfile = pyimport("mrcfile")

    if !isfile("BioTISR_Mitochondria.zip")
        @info "Dataset not found, downloading it..."
        Downloads.download(
            "https://zenodo.org/records/13843670/files/BioTISR_Mitochondria.zip",
        )
    end
    archive = ZipReader(read("BioTISR_Mitochondria.zip"))

    cell_id = "006"
    mrcdata =
        zip_readentry(archive, "BioTISR_Mitochondria/Cell_$cell_id/SIM_gt.mrc")
    write("biotisr-mitochondria-sim-gt-$cell_id.mrc", mrcdata)
    mrc = pyconvert(
        Array,
        mrcfile.read("biotisr-mitochondria-sim-gt-$cell_id.mrc"),
    )
    img_tensor = permutedims(mrc, (3, 2, 1))

    @save "../biotisr-mitochondria-sim-gt-$cell_id.jld2" img_tensor

    @info "JLD2 file successfully created." cell_id
end
