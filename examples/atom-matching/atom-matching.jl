using Revise
using PointCloudRegistration
using BioStructures
using Rotations
using GLMakie

atoms1 = collectatoms(retrievepdb("1iwo")["A"], backboneselector);
atoms2 = collectatoms(retrievepdb("1su4"), backboneselector);

source = PointCloud(stack(coords, atoms1))
target = PointCloud(stack(coords, atoms2))

motion_by_match = rigid_registration(
    source,
    target,
    GemanMcClureMM(batching = StochasticBatch(50));
    correspondences = MatchingLabels(atomname.(atoms1), atomname.(atoms2)),
)

motion_by_kc = rigid_registration(
    source,
    target,
    KernelCorrelationMM();
)

diff_motion = motion_by_kc ∘ inv(motion_by_match)
@show "diff" rad2deg(rotation_angle(diff_motion.linear))

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    tgl = Toggle(fig[2, 1]; tellwidth = false)
    rr_source = map(tgl.active) do active
        motion = ifelse(active, motion_by_match, motion_by_kc)
        return motion(source)
    end
    # colors1 = atomname2color.(atomname.(atoms1))
    # colors2 = atomname2color.(atomname.(atoms2))
    plot!(ax, rr_source; #= color = colors1 =#)
    plot!(ax, target; #= color = colors2 =#)
    fig
end

function atomname2color(n)
    if startswith(n, "O")
        :red
    elseif startswith(n, "C")
        :black
    elseif startswith(n, "H")
        :white
    elseif startswith(n, "N")
        :blue
    else
        :gray
    end
end
