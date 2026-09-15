using Revise
using PointCloudRegistration
using GLMakie
using GeometryBasics
using ColorTypes
using PubChemCrawler
using JSON
using DimensionalData
using StaticArrays
using LinearAlgebra
includet("impl.jl")

# compound_names = (source = "aspirin", target = "methyl salicylate")
compound_names = (source = "adenine", target = "adenosine")
# compound_names = (source = "cobalamin", target = "coenzyme b12")

cids = map(cached_get_cid, compound_names)
# cids = (source = 98285, target = 6305)
# cids = (source = 73415824, target = 6474320)

compounds = load_pubchem_data(cids);

correspondences_atomtypes = matching_labels(compounds.source.elements, compounds.target.elements)
correspondences_filtered =
    compatible_triangles(correspondences_atomtypes, compounds.source.pointcloud, compounds.target.pointcloud; deviation = 0.01)
@info "number of correspondences" length(correspondences_atomtypes.idcs) length(correspondences_filtered.idcs)

algorithms = [Kabsch(), GemanMcClureMM()];
corrs = [correspondences_atomtypes, correspondences_filtered];
alg_labels = ["Kabsch", "Geman-McClure"];
corrs_labels = ["matching elements", "compatible triangles"];

motions = [
    rigid_registration(
        compounds.source.pointcloud,
        compounds.target.pointcloud,
        alg;
        correspondences = corr,
    )
    for alg in algorithms, corr in corrs
]

let
    fig = Figure()
    Box(fig[0:2, 0:2]; color = :transparent, cornerradius = 20, strokecolor = :black)
    axs = [
        Axis3(
            fig[Tuple(ci)...];
            aspect = :data,
            protrusions = 0,
            alignmode = Outside(-30),
            viewmode = :stretch,
            elevation = deg2rad(120),
            azimuth = deg2rad(70),
            # elevation = deg2rad(150),
            # azimuth = deg2rad(70),
        )
        for ci in CartesianIndices(motions)
    ]
    hidedecorations!.(axs)
    hidespines!.(axs)
    Label(fig[0, 1], "matching elements"; font = :bold, tellwidth = false)
    Label(fig[0, 2], "compatible triangles"; font = :bold, tellwidth = false)
    Label(fig[1, 0], "Kabsch"; font = :bold, tellheight = false, rotation = deg2rad(90))
    Label(fig[2, 0], "Geman-McClure"; font = :bold, tellheight = false, rotation = deg2rad(90))
    target_color_change = function(c)
        (; h, s, l) = HSL(c)
        # s /= 3
        l *= 1.5
        HSL(h, s, l)
    end
    for ci in CartesianIndices(motions)
        ax = axs[ci]
        m = motions[ci]
        moved_source = m(compounds.source.pointcloud)
        plot_molecule!(ax, Molecule(moved_source, compounds.source.elements, compounds.source.bonds); const_color = colorant"#0074d9")
        plot_molecule!(ax, compounds.target; color_change = target_color_change, const_color = colorant"#ff4136")
    end
    colgap!(fig.layout, 0)
    rowgap!(fig.layout, 0)
    resize_to_layout!(fig)
    fig
    save("$(compound_names.source)-$(compound_names.target).png", fig; px_per_unit = 2)
end

