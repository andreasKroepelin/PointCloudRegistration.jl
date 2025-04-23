function show_restarts(al, rows, cols, d)
    mm_logs = filter(log -> log.id == :mm, al.logs);
    mm_logs_per_restart = [filter(log -> log.restart == rows * (col - 1) + row - 1, mm_logs) for row in 1:rows, col in 1:cols];
    max_kc = maximum(log -> isnan(log.kc) ? -Inf : log.kc, mm_logs)
    @info "collected data" max_kc
    fig = Figure()
    slider = Slider(fig[2, 1][1, 1]; range = 1:maximum(length, mm_logs_per_restart))
    my_colormap = Makie.PlotUtils.cgrad([Makie.Colors.alphacolor(colorant"tomato", 0.), colorant"tomato"])
    @info "setup figure done"
    main_axes = []
    kc_axes = []
    for row in 1:rows, col in 1:cols
        @info "creating plot" row col
        logs = mm_logs_per_restart[row, col]
        # kcs_per_invΔ = Dict(
        #     invΔ => [invΔ * log.kc for log in logs if log.grid.invΔ == invΔ]
        #     for invΔ in unique([log.grid.invΔ for log in logs])
        # )
        curr_log = @lift logs[min($(slider.value), lastindex(logs))]
        curr_rotation = @lift ($curr_log).rotation
        curr_translation = @lift ($curr_log).translation
        curr_trY = @lift $curr_rotation * Y .+ $curr_translation
        curr_target_kde = @lift ($curr_log).target_kde
        curr_grid = @lift ($curr_log).grid
        # curr_kcs = @lift kcs_per_invΔ[($curr_grid).invΔ]
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
            heatmap!(ax, curr_domains..., curr_target_kde, colormap=my_colormap)
            push!(main_axes, ax)
        elseif d == 3
            ax = Axis3(fig[1, 1][row, col]; aspect = :equal, title)
            volume!(ax, curr_intervals..., curr_target_kde, colormap=my_colormap)
        end
        hidedecorations!(ax)
        scatter!(ax, curr_trY; color = :teal)

        kc_ax = Axis(fig[1, 1][row, col][2, 1]) # width = Relative(.8), height = Relative(.2), halign = .1, valign = .1)
        push!(kc_axes, kc_ax)
        # # lines!(kc_ax, curr_kcs)
        lines!(kc_ax, [log.kc for log in logs])
        vlines!(kc_ax, slider.value, linestyle = :dash)
        hlines!(kc_ax, max_kc, linestyle = :dot)
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
    end

    # kc_ax = Axis(fig[2, 1][1, 2], height = Fixed(400))
    # for row in 1:rows, col in 1:cols
    #     logs = mm_logs_per_restart[row, col]
    #     lines!(kc_ax, [log.kc for log in logs])
    # end

    fig
end
