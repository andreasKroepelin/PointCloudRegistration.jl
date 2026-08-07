using Revise
using BenchmarkTools
using Makie
import GLMakie
import CairoMakie
using Statistics
using Dates
using StaticArrays
using HybridArrays
import PointCloudRegistration as PCReg
import CoordinateTransformations as CT
import BioStructures as BioS
import Kabsch as K

# Xpc, Ypc = PCReg.Assets.load_1ake_A_4ake_A()
# X, Y, = stack.((Xpc.points, Ypc.points))
# X = [X X X X]
# Y = [Y Y Y Y]
X = randn(3, 1000)
Y = randn(3, 1000)
Xs, Ys = HybridMatrix{3, StaticArrays.Dynamic()}.((X, Y))

bms = (
    pcreg = @benchmark(PCReg.rigid_registration($Y, $X, PCReg.Kabsch())),
    # pcreg_gmc = @benchmark(PCReg.rigid_gmc($Y, $X)),
    # pcreg_pc = @benchmark(PCReg.rigid_rmsd($Ypc, $Xpc)),
    ct = @benchmark(CT.kabsch($Y => $X)),
    bios = @benchmark(BioS.Transformation($Y, $X)),
    k = @benchmark(K.kabsch($Y, $X)),
)

labels = (
    pcreg = rich("PointCloudRegistration.jl"; font = :bold),
    pcreg_pc = "PCReg.jl w/o\nprestackaration",
    pcreg_gmc = "PCReg.jl GMC",
    ct = "CoordinateTransformations.jl",
    bios = "BioStructures.jl",
    k = "Kabsch.jl",
)

let
    # fig = Figure(size = (400, 250))
    fig = Figure()
    kys = collect(keys(bms))
    ax = Axis(fig[1, 1], xticks = (eachindex(kys), [labels[k] for k in kys]))
    ylims!(ax, (0, nothing))
    # all_times = mapreduce(key -> bms[key].times, vcat, kys)
    # all_categories = mapreduce(((i, key),) -> fill(i, length(bms[key].times)), vcat, enumerate(kys))
    all_times = Float64[]
    all_categories = Int[]
    for (i, key) in enumerate(kys)
        (; times) = bms[key]
        scatter!(ax, [i], [mean(times) / 1e3])
        if key != :pcreg
            slowest = quantile(times, .99)
            times = filter(t -> t < slowest, times)
        end
        append!(all_times, times)
        append!(all_categories, fill(i, length(times)))
        # q = map((.05, .5, .95)) do p
        #     quantile(times, p) |> round |> Nanosecond
        # end
        # rangebars!(ax, [i], [q[1]], [q[3]]; linewidth = 3)
        # scatter!(ax, [i], [q[2]]; markersize = 15)
    end
    boxplot!(
        ax,
        all_categories,
        all_times ./ 1e3;
        width = 1.0,
        # color = indexin(all_categories, unique(all_categories))
    )
    GLMakie.activate!(); display(fig)
    # CairoMakie.activate!(pdf_version = "1.5");
    # save("../paper/bioinformatics/src/img/kabsch-benchmark.pdf", fig)
    # save(expanduser("~/kabsch-benchmark.pdf"), fig)
end

pc_dim = 3
pc_size = 1000
num_pcs = 10_000
timings = let
    sources = [rand(pc_dim, pc_size) for _ in 1:num_pcs];
    target = rand(pc_dim, pc_size);
    pcreg = @timed for source in sources
        PCReg.rigid_registration(source, target, PCReg.Kabsch())
    end
    ct = @timed for source in sources
        CT.kabsch(source => target)
    end
    bios = @timed for source in sources
        BioS.Transformation(source, target)
    end
    k = @timed for source in sources
        K.kabsch(source, target)
    end
    (; pcreg, bios, k, ct)
end

nord_theme = Theme(
    fontsize = 20,
    fonts = (
        regular = "Atkinson Hyperlegible Next",
        bold = "Atkinson Hyperlegible Next Bold",
    ),
    backgroundcolor = "#3b4252",
    linecolor = "#eceff4",
    textcolor = "#eceff4",
    pathstrokecolor = "#eceff4",
    markerstrokecolor = "#eceff4",
    markercolor = "#eceff4",
)

with_theme(nord_theme) do
    fig = Figure(; size = (700, 500), fontsize = 20, fonts = (; regular = "Atkinson Hyperlegible Next", bold = "Atkinson Hyperlegible Next Bold"))
    kys = collect(keys(timings))
    sort!(kys; by = key -> timings[key].time)
    ax = Axis(fig[1, 1], yticks = (eachindex(kys), [labels[k] for k in kys]), xtickformat = "{:.1f} s", backgroundcolor = "#3b4252", xgridcolor = "#4c566a", title = "Kabsch registration of $num_pcs point clouds in $pc_dim dimensions \n with $pc_size points each")
    hideydecorations!(ax)
    hidespines!(ax)
    all_times = Float64[]
    all_categories = Int[]
    all_groups = Int[]
    for (i, key) in enumerate(kys)
        (; time, gctime) = timings[key]
        append!(all_times, [gctime, time - gctime])
        append!(all_categories, [i, i])
        append!(all_groups, [1, 2])
    end
    ylims!(ax, (0.5, last(all_categories) + 2))
    barplot!(
        ax,
        all_categories,
        all_times;
        direction = :x,
        stack = all_groups,
        color = all_groups,
        colormap = ["#d08770", "#5e81ac"]
    )
    bracket!(ax, 0, last(all_categories) + 1, timings[last(kys)].time, last(all_categories) + 1; text = "total time", linewidth = 3)
    bracket!(ax, 0, last(all_categories) + .4, timings[last(kys)].gctime, last(all_categories) + .4; text = "GC time", linewidth = 3)
    for (i, key) in enumerate(kys)
        # text!(ax, Point(maximum(k -> timings[k].time, kys), i); text = labels[key], align = (:right, :center))
        text!(ax, Point(0, i); text = labels[key], align = (:left, :center), offset = (10, 0))
    end
    # GLMakie.activate!(); display(fig)
    CairoMakie.activate!(pdf_version = "1.5");
    # save("../paper/bioinformatics/src/img/kabsch-benchmark.pdf", fig)
    save("kabsch-benchmark.svg", fig)
end






@btime PCReg.rigid_gmc($Y, $X; scale = .1)
