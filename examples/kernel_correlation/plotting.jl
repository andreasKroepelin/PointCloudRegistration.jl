function show_restarts(al, rows, cols, prepd_target)
    (; axis_aligning_rotation, target) = prepd_target
    d = size(target, 1)
    mm_logs = filter(log -> log.id == :mm, al.logs)
    mm_logs_per_restart = [
        filter(log -> log.restart == rows * (col - 1) + row - 1, mm_logs)
        for row in 1:rows, col in 1:cols
    ]
    min_kc, max_kc = extrema(log.kc for log in mm_logs if !isnan(log.kc))
    @info "collected data" min_kc, max_kc
    fig = Figure()
    display(fig)
    slider =
        Slider(fig[2, 1][1, 1]; range = 1:maximum(length, mm_logs_per_restart))
    my_colormap = Makie.PlotUtils.cgrad([
        Makie.Colors.alphacolor(colorant"tomato", 0.0),
        colorant"tomato",
    ])
    @info "setup figure done"
    main_axes = []
    kc_axes = []
    for row in 1:rows, col in 1:cols
        @info "creating plot" row col
        logs = mm_logs_per_restart[row, col]
        curr_log::Observable{NamedTuple} = @lift logs[min($(slider.value), lastindex(logs))]
        curr_rotation = @lift ($curr_log).rotation
        curr_translation = @lift ($curr_log).translation
        curr_trY = @lift $curr_rotation * Y .+ $curr_translation
        curr_target_kde = @lift ($curr_log).target_kde
        curr_grid = @lift ($curr_log).grid
        # curr_kcs = @lift kcs_per_invΔ[($curr_grid).invΔ]
        curr_kc = @lift [($curr_log).kc]
        if d == 2
            curr_domains = map((1, 2)) do i
                @lift PointCloudRegistration.domains($curr_grid)[i]
            end
        elseif d == 3
            curr_intervals = map((1, 2, 3)) do i
                lift(curr_grid) do grid
                    dom = PointCloudRegistration.domains(grid)[i]
                    Makie.ClosedInterval(extrema(dom)...)
                end
            end
        end
        title = @lift string(
            "grid cells = ",
            prod(($curr_grid).size),
            " ; iter = ",
            ($curr_log).iter,
        )
        if d == 2
            ax = Axis(fig[1, 1][row, col][1, 1]; autolimitaspect = 1, title)
            heatmap!(
                ax,
                curr_domains...,
                curr_target_kde;
                colormap = my_colormap,
            )
            push!(main_axes, ax)
        elseif d == 3
            ax = Axis3(fig[1, 1][row, col][1, 1]; aspect = :data, protrusions = 0, title)
            # volume!(
            #     ax,
            #     curr_intervals...,
            #     curr_target_kde;
            #     colormap = my_colormap,
            # )
            push!(main_axes, ax)
        end
        hidedecorations!(ax)
        meshscatter!(ax, curr_trY; color = :teal)
        meshscatter!(ax, target.points; color = :tomato)

        kc_ax = Axis(fig[1, 1][row, col][1, 2], width = 20) # width = Relative(.8), height = Relative(.2), halign = .1, valign = .1)
        ylims!(kc_ax, (min_kc, max_kc))
        xlims!(kc_ax, (.5, 1.5))
        hidexdecorations!(kc_ax)
        push!(kc_axes, kc_ax)
        barplot!(kc_ax, curr_kc, fillto = min_kc, width = 1, gap = 0)
        # lines!(kc_ax, curr_kcs)
        # lines!(kc_ax, [log.kc for log in logs])
        # vlines!(kc_ax, slider.value; linestyle = :dash)
        # hlines!(kc_ax, max_kc; linestyle = :dot)
    end
    ax, other_axes... = kc_axes
    for other_ax in other_axes
        linkyaxes!(ax, other_ax)
    end
    if d == 2
        ax, other_axes... = main_axes
        for other_ax in other_axes
            linkaxes!(ax, other_ax)
        end
    elseif d == 3
        ax, other_axes... = main_axes
        on(ax.azimuth) do az
            for other_ax in other_axes
                other_ax.azimuth = az
            end
        end
        on(ax.elevation) do el
            for other_ax in other_axes
                other_ax.elevation = el
            end
        end
    end
    colgap!(contents(fig[1, 1])[1], 60)

    # kc_ax = Axis(fig[2, 1][1, 2], height = Fixed(400))
    # for row in 1:rows, col in 1:cols
    #     logs = mm_logs_per_restart[row, col]
    #     lines!(kc_ax, [log.kc for log in logs])
    # end

    fig
end

function side_by_side(source, target, transformation)
    source = transformation(PointCloud(source))
    fig = Figure()
    slider = Slider(fig[2, 1:2], range = .01:.01:2)
    ax_src = Axis3(fig[1, 1], title = "source", aspect = :data)
    ax_trg = Axis3(fig[1, 2], title = "target", aspect = :data)
    meshscatter!(ax_src, source.points, color = axes(source.points, 2), markersize = slider.value)
    meshscatter!(ax_trg, target.points, color = axes(target.points, 2), markersize = slider.value)

    on(ax_src.azimuth) do az
        ax_trg.azimuth = az
    end
    on(ax_src.elevation) do el
        ax_trg.elevation = el
    end

    fig
end
