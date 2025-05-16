using Revise
using Makie, GLMakie
using BioStructures
using PointCloudRegistration
using Printf

groel = retrievepdb("1oel"; dir = tempdir())
subunit = groel["A"]

target = coordarray([groel["B"], groel["D"], groel["F"]], calphaselector)
source = PointCloud(coordarray(subunit, calphaselector))

prepd_target = prepare_target_kc(target, scale = 1., annealing = 5);

twcs = register_kc(source, prepd_target; accumulator = AllTransformations, restarts = 100_000);

means = stack(twcs) do twc
    twc.transformation(source.mean)[1:2]
end
kcs = map(twc -> -twc.cost, twcs);
hist(kcs, bins = 100; axis = (; yscale = log10))
best_twcs = twcs[kcs .>= 0.95 .* maximum(kcs)];
# best_twcs = sort(twcs, by = twc -> twc.cost)[1:100];
length(best_twcs)

fig = Figure()
sl = Slider(fig[2, 1], range = range(extrema(kcs)..., length = 100))
Label(fig[2, 2], @lift(@sprintf "%1.4e" $(sl.value)))
ax = Axis(fig[1, 1:2], autolimitaspect = 1)
scatter!(ax, target[1:2, :], alpha = .3)
colors = map(sl.value) do threshold
    kcs .>= threshold
end
scatter!(ax, means, color = colors, alpha = .5, colormap = [(:gray90, .01), :red])


# best = register_kc(source, prepd_target; accumulator = BestTransformation, restarts = 10_000);

fig = Figure()
ax = Axis(fig[1, 1], autolimitaspect = 1)
sl = Slider(fig[2, 1], range = eachindex(best_twcs))
to_plot = @lift collect(best_twcs[$(sl.value)].transformation(source))[1:2, :]
scatter!(ax, target[1:2, :], markersize = 10)
scatter!(ax, to_plot, markersize = 10)
