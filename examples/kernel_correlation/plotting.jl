function show_restarts(al, rows, cols, d)
    mm_logs = filter(log -> log.id == :mm, al.logs);
    mm_logs_per_restart = [filter(log -> log.restart == rows * (col - 1) + row - 1, mm_logs) for row in 1:rows, col in 1:cols];
    @info "collected data"
    fig = Figure()
    slider = Slider(fig[2, 1]; range = 1:maximum(length, mm_logs_per_restart))
    my_colormap = Makie.PlotUtils.cgrad([Makie.Colors.alphacolor(colorant"tomato", 0.), colorant"tomato"])
    @info "setup figure done"
    for row in 1:rows, col in 1:cols
        @info "creating plot" row col
        logs = mm_logs_per_restart[row, col]
        curr_log = @lift logs[min($(slider.value), lastindex(logs))]
        curr_rotation = @lift ($curr_log).rotation
        curr_translation = @lift ($curr_log).translation
        curr_trY = @lift $curr_rotation * Y .+ $curr_translation
        curr_target_kde = @lift ($curr_log).target_kde
        curr_grid = @lift ($curr_log).grid
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
            ax = Axis(fig[1, 1][row, col]; autolimitaspect = 1, title)
            heatmap!(ax, curr_domains..., curr_target_kde, colormap=my_colormap)
        elseif d == 3
            ax = Axis3(fig[1, 1][row, col]; aspect = :equal, title)
            volume!(ax, curr_intervals..., curr_target_kde, colormap=my_colormap)
        end
        hidedecorations!(ax)
        scatter!(ax, curr_trY; color = :teal)
    end
    if d == 2
        ax, other_axes... = contents(contents(fig[1, 1])[1]);
        for other_ax in other_axes
            linkaxes!(ax, other_ax)
        end
    end

    fig
end
