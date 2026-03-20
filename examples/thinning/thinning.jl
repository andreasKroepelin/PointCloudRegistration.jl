using Revise
using PointCloudRegistration
using Makie
import GLMakie, CairoMakie
using LinearAlgebra
using BenchmarkTools

unit_square = 10 .* rand(2, 10000)

inshape(x) = any(
    center -> norm(x - center) < 1,
    ([3, 3], [2, 5], [4, 3], [4, 2], [3, 6], [3, 4])
)

pc = filter(inshape, eachcol(unit_square)) |> stack |> PointCloud

GLMakie.activate!()
plot(pc)

b1 = @btimed thin_to_grid($pc, .4)
b2 = @btimed thin_to_distance($pc, .4)
b3 = @btimed thin_to_number($pc, $(length(b2.value.points)))

let
    pcs = [pc, [b.value for b in [b1, b2, b3]]...]
    for (pc, name, sizefactor) in zip(pcs, ["original", "grid", "distance", "number"], [.05, fill(.015, 3)...])
        fig = Figure(size = (150, 290))
        ax = Axis(fig[1, 1]; aspect = DataAspect())
        hidedecorations!(ax)
        hidespines!(ax)
        plot!(ax, pc; sizefactor)
        resize_to_layout!(fig)
        # GLMakie.activate!(); wait(display(fig));
        CairoMakie.activate!(); save(expanduser("~/Pictures/thinning-$name.svg"), fig)
    end
end
