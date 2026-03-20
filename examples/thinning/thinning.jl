using Revise
using PointCloudRegistration
using Makie
import GLMakie, CairoMakie
using LinearAlgebra
using BenchmarkTools

unit_square = rand(2, 10000)

inshape(x) = any(
    center -> norm(x - center) < .1,
    ([.3, .3], [.2, .5], [.4, .3], [.4, .2], [.3, .6], [.3, .4])
)

pc = filter(inshape, eachcol(unit_square)) |> stack |> PointCloud

GLMakie.activate!()
plot(pc)

pc1 = @btime thin_to_grid(pc, .04)
pc2 = @btime thin_to_distance(pc, .04)
pc3 = @btime thin_to_number(pc, 100)

let
    fig = Figure()
    for (i, thinned_pc) in enumerate([pc, pc1, pc2, pc3])
        ax = Axis(fig[1, i]; aspect = DataAspect())
        plot!(ax, thinned_pc; sizefactor = .002)
    end
    fig
end
