using Revise
using BenchmarkTools
using GLMakie
using Statistics
using Dates
using StaticArrays
using HybridArrays
import PointCloudRegistration as PCReg
import CoordinateTransformations as CT
import BioStructures as BioS
import Kabsch as K

Xpc, Ypc = PCReg.Assets.load_1ake_A_4ake_A()
X, Y, = collect.((Xpc, Ypc))
Xs, Ys = HybridMatrix{3, StaticArrays.Dynamic()}.((X, Y))

bms = (
    pcreg = @benchmark(PCReg.register_rmsd($Y, $X)),
    pcreg_pc = @benchmark(PCReg.register_rmsd($Ypc, $Xpc)),
    ct = @benchmark(CT.kabsch($Y => $X)),
    bios = @benchmark(BioS.Transformation($Y, $X)),
    k = @benchmark(K.kabsch($Y, $X)),
)
labels = (
    pcreg = "PCReg.jl",
    pcreg_pc = "PCReg.jl w/o\npreparation",
    ct = "CoordTr.jl",
    bios = "BioStr.jl",
    k = "Kabsch.jl",
)

let
    fig = Figure()
    kys = collect(keys(bms))
    ax = Axis(fig[1, 1], xticks = (eachindex(kys), [labels[k] for k in kys]))
    ylims!(ax, (0, nothing))
    for (i, key) in enumerate(kys)
        (; times) = bms[key]
        q = map((.25, .5, .75)) do p
            quantile(times, p) |> round |> Nanosecond
        end
        rangebars!(ax, [i], [q[1]], [q[3]])
        scatter!(ax, [i], [q[2]]; markersize = 10)
    end
    fig
end

@btime PCReg.register_gmc($Y, $X)
