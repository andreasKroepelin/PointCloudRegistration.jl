function plot_data(data)
    smm_colors = Dict(
        0 => colorant"#85144b",
        10 => colorant"#0074D9",
        50 => colorant"#39CCCC",
        100 => colorant"#7FDBFF",
    )
    fig = Figure()
    ax_sc = Axis(fig[0, 1]; ylabel = "annealing scales", yticks = unique(reduce(vcat, data.scale)), ytickformat = "{:.2f}")
    ax_it = Axis(fig[1, 1]; ylabel = "iterations", yticks = unique(data.iterations))
    ax_su = Axis(fig[2, 1]; ylabel = "number of restarts\nfor 1 % failure rate", xlabel = "strategies")
    Legend(
        fig[2, 2],
        [
            PolyElement(; color = smm_colors[n], strokewidth = 0)
            for n in [0, 10, 50, 100]
        ],
        ["without SMM", "with SMM (10)", "with SMM (50)", "with SMM (100)"];
        halign = :left,
        valign = :bottom,
        # tellwidth = false,
        tellheight = false,
    )
    linkxaxes!(ax_sc, ax_su)
    linkxaxes!(ax_sc, ax_it)
    hidexdecorations!(ax_sc)
    hidexdecorations!(ax_it)
    hidexdecorations!(ax_su; label = false)
    sorted_idcs = sortperm(data.success_rate; rev = true)
    stripes_lo = (1:2:length(data)) .- 0.5
    stripes_hi = (1:2:length(data)) .+ 0.5
    for ax in (ax_sc, ax_it, ax_su)
        vspan!(
            ax,
            stripes_lo,
            stripes_hi;
            color = (:gray, 0.1),
            inspectable = false,
        )
    end
    plot_idx = 0
    for i in sorted_idcs
        scale = data[i].scale
        plot_idx += 1
        x = fill(plot_idx, length(scale))
        points = mapreduce(vcat, scale) do s
            [Point(plot_idx - 0.2, s), Point(plot_idx + 0.2, s)]
        end
        color = smm_colors[data[i].smm_count]
        linesegments!(ax_sc, points; color, linewidth = 4, linecap = :round)
        scatter!(ax_it, Point(plot_idx, data[i].iterations); color)
        num_trials = data.necessary_restarts[i]
        if isfinite(num_trials)
            barplot!(ax_su, Point(plot_idx, num_trials); color)
        else
            text!(
                ax_su,
                Point(plot_idx, 0);
                text = "?",
                align = (:center, :bottom),
            )
        end
        # barplot!(ax_su, Point(plot_idx, success_rates[i]); color)
    end
    rowgap!(fig.layout, 2)
    resize_to_layout!(fig)
    # DataInspector(fig)
    fig
end

