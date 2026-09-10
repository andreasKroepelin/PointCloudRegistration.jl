using Revise
using PointCloudRegistration
using GLMakie
using PubChemCrawler
using BiochemicalAlgorithms
using JSON
includet("impl.jl")

# compound_names = (source = "aspirin", target = "methyl salicylate")
compound_names = (source = "adenine", target = "adenosine")

cids = map(cached_get_cid, compound_names)
# cids = (source = 98285, target = 6305)

compounds = load_pubchem_data(cids);
atomtypes = map(c -> atoms(c).element, compounds)
colors = map(atomtypes2colors, atomtypes)
coords = map(c -> atoms(c).r, compounds)
bs = map(get_bonds, compounds)
pcs = map(PointCloud, coords);

correspondences_atomtypes = matching_labels(atomtypes.source, atomtypes.target)
correspondences_filtered =
    compatible_triangles(correspondences_atomtypes, pcs.source, pcs.target; deviation = 0.01)
@info "number of correspondences" length(correspondences_atomtypes.idcs) length(correspondences_filtered.idcs)

motion_naive = rigid_registration(pcs.source, pcs.target, NoFlip())
motion_atoms = rigid_registration(
    pcs.source,
    pcs.target,
    NoFlip();
    correspondences = correspondences_atomtypes,
)
motion_filtered = rigid_registration(
    pcs.source,
    pcs.target,
    NoFlip();
    correspondences = correspondences_filtered,
)

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    menu = Menu(
        fig[2, 1];
        options = [
            ("naive", motion_naive),
            ("atom types", motion_atoms),
            ("triangles", motion_filtered),
        ]
    )
    moved_source = map(motion -> motion(pcs.source), menu.selection)
    linesegments!(
        ax,
        @lift([ ($moved_source[i].coords, $moved_source[j].coords)
          for (i, j) in bs.source ]);
        color = :orange,
        linewidth = 5,
    )
    linesegments!(
        ax,
        [ (pcs.target[i].coords, pcs.target[j].coords)
          for (i, j) in bs.target ];
        color = :aqua,
        linewidth = 5,
    )
    markers = (
        source = Sphere(Point3(0.0), 0.5),
        target = Rect3(Point3(-0.5), Point3(1)),
    )
    plot!(ax, moved_source; marker = markers.source, sizefactor = .5, color = colors.source)
    plot!(ax, pcs.target; marker = markers.target, sizefactor = .5, color = colors.target)
    fig
end

rr_system = System{Float32}()
rr_source_compound = Molecule(rr_system)
rr_source_atoms = []
for (i, c, t) in zip(1:length(pcs.source), points(motion_filtered(pcs.source)), atomtypes.source)
    push!(rr_source_atoms, Atom(rr_source_compound, i, t; r = c))
end
for ((i, j), b) in zip(bs.source, bonds(compounds.source))
    Bond(rr_source_compound, rr_source_atoms[i].idx, rr_source_atoms[j].idx, b.order)
end
rr_target_compound = Molecule(rr_system)
rr_target_atoms = []
for (i, at) in enumerate(atoms(compounds.target))
    push!(rr_target_atoms, Atom(rr_target_compound, i, at.element; at.r))
end
for ((i, j), b) in zip(bs.target, bonds(compounds.target))
    Bond(rr_target_compound, rr_target_atoms[i].idx, rr_target_atoms[j].idx, b.order)
end
write_sdfile("rr-$(compound_names.source)-$(compound_names.target).sdf", rr_system)
