using Revise
using Makie
import GLMakie, CairoMakie
using BioStructures
using PointCloudRegistration
using Printf
using StatsBase
using TerminalLoggers
using Logging

global_logger(TerminalLogger());
# CairoMakie.activate!(; type = "png")
GLMakie.activate!()

groel = retrievepdb("1oel"; dir = tempdir())
subunit = groel["A"]

subunit_means = stack('A':'G') do chain_id
    chain = groel[chain_id]
    coords = coordarray(chain, calphaselector)
    vec(mean(coords; dims = 2))
end

target = coordarray(groel, calphaselector)
source = PointCloud(coordarray(subunit, calphaselector))

maxscale = 0.7 * sqrt(maximum(source.coveigvals))
scales = logrange(maxscale, 5.0; length = 5)
prepd_target = prepare_target_kc(target; scale = scales);

twcs = register_kc(
    source,
    prepd_target;
    accumulator = AllTransformations,
    restarts = 50_000,
);

means = stack(twcs) do twc
    twc.transformation(source.mean)[1:2]
end
kcs = map(twc -> -twc.cost, twcs);
min_kc, max_kc = extrema(kcs)
hist(kcs; bins = 100, axis = (; yscale = log10))
best_mask = kcs .>= 0.5 .* max_kc;
best_twcs = @view twcs[best_mask];
# best_twcs = sort(twcs, by = twc -> twc.cost)[1:100];
length(best_twcs)

fig = Figure()
ax = Axis(fig[1, 1]; autolimitaspect = 1)
ax3 = Axis3(fig[1, 2])
ctrl_gl = GridLayout(fig[2, 1:2])
sl = Slider(ctrl_gl[1, 1]; range = 0:0.01:1)
Label(ctrl_gl[1, 2], @lift(@sprintf "%1.2f" $(sl.value)))
selected_idcs = map(sl.value) do threshold
    kcs .- min_kc .>= threshold * (max_kc - min_kc)
end
selected_means = @lift @view means[:, $selected_idcs]
selected_kcs = @lift @view kcs[$selected_idcs]
selected_histogram =
    @lift fit(Histogram, Tuple(eachrow($selected_means)), nbins = 100).weights
# scatter!(ax, target[1:2, :]; strokewidth = 1, strokecolor = :white, color = :, alpha = 1)
surface!(ax3, selected_histogram; colormap = :blues)
scatter!(
    ax,
    selected_means;
    color = selected_kcs,
    colorscale = log10,
    colorrange = (min_kc, max_kc),
    colormap = :binary,
    alpha = 0.5,
    markersize = 20,
)

function density_of_optima(threshold)
    fig = Figure()
    ax3 = Axis3(fig[1, 1])
    selected_idcs = kcs .- min_kc .>= threshold * (max_kc - min_kc)
    selected_means = @view means[:, selected_idcs]
    selected_histogram =
        fit(Histogram, Tuple(eachrow(selected_means)); nbins = 100).weights
    surface!(ax3, selected_histogram; colormap = :blues)
    fig
end

density_of_optima(0.1)

fig = Figure()
ax = Axis(fig[1, 1])
tricontourf!(ax, eachrow(means[:, best_mask])..., kcs[best_mask], colorscale = log10)


# best = register_kc(source, prepd_target; accumulator = BestTransformation, restarts = 10_000);

fig = Figure()
ax = Axis(fig[1, 1]; autolimitaspect = 1)
sl = Slider(fig[2, 1]; range = eachindex(best_twcs))
to_plot = @lift collect(best_twcs[$(sl.value)].transformation(source))[1:2, :]
scatter!(ax, target[1:2, :]; markersize = 10)
scatter!(ax, to_plot; markersize = 10)
