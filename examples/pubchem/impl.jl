struct Bond
    from::Int
    to::Int
    order::Int
end

struct Molecule
    pointcloud::PointCloud
    elements::Vector
    bonds::Vector{Bond}
end


function cached_get_cid(name)
    cachefile = "cache-cid-$name"
    if !isfile(cachefile)
        @info "missing cache file for $name, calling PubChem API"
        cid = get_cid(; name)
        write(cachefile, cid)
    end
    return read(cachefile, Int64)
end

function load_pubchem_data(cids; record_type = "3d")
    all_cids = Int64[]
    # for file in readdir()
    #     startswith(file, "cache-cid-") || continue
    #     push!(all_cids, read(file, Int64))
    # end
    append!(all_cids, values(cids))
    unique!(all_cids)
    needs_download = true
    if isfile("cache-pubchem")
        json = JSON.parse(read("cache-pubchem", String))
        if all(cid -> find_compound(json, cid) !== nothing, all_cids)
            needs_download = false
        end
    end
    if needs_download
        @info "Downloading PubChem data for $(length(all_cids)) compounds."
        pubchem_res = get_for_cids(all_cids; output = "JSON", record_type)
        write("cache-pubchem", pubchem_res)
        json = JSON.parse(read("cache-pubchem", String))
    end
    return map(cid -> find_compound(json, cid), cids)
end

function find_compound(json, cid)
    i = findfirst(c -> c.id.id.cid == cid, json.PC_Compounds)
    isnothing(i) && return nothing
    c =  json.PC_Compounds[i]
    at = c.atoms
    elements = at.element
    b = c.bonds
    bonds = map(Bond, b.aid1, b.aid2, b.order)
    cnf = c.coords[1].conformers[1]
    z = :z in keys(cnf) ? cnf.z : zeros(length(elements))
    crds = hcat(cnf.x, cnf.y, z) |> transpose .|> float
    pointcloud = PointCloud(crds)
    return Molecule(pointcloud, elements, bonds)
end

function elementcolors(m)
    map(m.elements) do e
        c = if e == 1 # H
            colorant"#eee"
        elseif e == 6 # C
            colorant"#222"
        elseif e == 7 # N
            colorant"#0074d9"
        elseif e == 8 # O
            colorant"#ff4136"
        else
            :gray
        end
        Makie.to_color(c)
    end
end

function atomradii(m)
    rs = [0.31, 0.28, 1.28, 0.96, 0.84, 0.73, 0.71, 0.66, 0.57]
    return map(e -> e in eachindex(rs) ? rs[e] : 0.1, m.elements)
end

function degrees(m)
    map(1:length(m.bonds)) do i
        count(m.bonds) do bond
            bond.from == i || bond.to == i
        end
    end
end

function neighbors(m, i)
    n = Int[]
    for bond in m.bonds
        if bond.from == i
            push!(n, bond.to)
        elseif bond.to == i
            push!(n, bond.from)
        end
    end
    return n
end


# [NOTE] The following is based on code in MolecularGraph.jl.
# (MIT License, Seiji Matsuoka and contributors)

function plot_molecule!(ax, mol::Molecule; color_change = identity, const_color = nothing)
    if const_color === nothing
        clr = color_change.(elementcolors(mol))
    else
        clr = fill(const_color, length(mol.elements))
    end
    dgr = degrees(mol)
    # meshscatter!(
    #     ax,
    #     points(mol.pointcloud);
    #     color = clr,
    #     markersize = [bonddiameter for _ in clr],
    #     # markersize = 0.8 .* atomradii(mol),
    # )
    for bond in mol.bonds
        (; from, to, order) = bond
        pos1 = mol.pointcloud[from].coords
        pos2 = mol.pointcloud[to].coords
        bonddiameter = if order == 1
            if mol.elements[from] == 1 || mol.elements[to] == 1
                0.1
            else
                0.2
            end
        else
            0.1
        end
        normaldir = SA[0.0, 0.0, 1.0]
        if order > 1
            ng1, ng2 = dgr[from], dgr[to]
            # determine the plane for double bonds
            if ng1 == 3
                neighs = filter(!=(to), neighbors(mol, from))
                @assert length(neighs) == 2
                npos1 = mol.pointcloud[neighs[1]].coords
                npos2 = mol.pointcloud[neighs[2]].coords
                normaldir = cross(npos1, npos2)
            elseif ng2 == 3
                neighs = filter(!=(from), neighbors(mol, to))
                @assert length(neighs) == 2
                npos1 = mol.pointcloud[neighs[1]].coords
                npos2 = mol.pointcloud[neighs[2]].coords
                normaldir = cross(npos1, npos2)
            end
        end
        sepdir = normalize(cross(normaldir, pos2 - pos1))
        dists = (bonddiameter * 2.5) .* collect((-0.5 * (order - 1)):(0.5 * (order - 1)))
        for dist in dists
            dvec = dist * sepdir
            p1, p2 = pos1 + dvec, pos2 + dvec
            if mol.elements[from] == mol.elements[to]
                cyl = Cylinder(Point3(p1), Point3(p2), bonddiameter)
                sph1 = Sphere(Point3(p1), bonddiameter)
                sph2 = Sphere(Point3(p2), bonddiameter)
                mesh!(ax, cyl; color = clr[from])
                mesh!(ax, sph1; color = clr[from])
                mesh!(ax, sph2; color = clr[from])
            else
                midpoint = 0.5 * (p1 + p2)
                pm = Point(midpoint)
                cyl1 = Cylinder(Point3(p1), pm, bonddiameter)
                cyl2 = Cylinder(Point3(p2), pm, bonddiameter)
                sph1 = Sphere(Point3(p1), bonddiameter)
                sph2 = Sphere(Point3(p2), bonddiameter)
                mesh!(ax, cyl1; color = clr[from])
                mesh!(ax, cyl2; color = clr[to])
                mesh!(ax, sph1; color = clr[from])
                mesh!(ax, sph2; color = clr[to])
            end
        end
    end
end

# function drawbond!(
#         f, mol::SimpleMolGraph, e, crds, col, syms, nbrs;
#         bonddiameter=DEFAULT_BOND_DIAMETER, multiplebonds=false, kwargs...)
#     order = multiplebonds ? bond_order(mol[e]) : 1
#     atomidx1, atomidx2 = e.src, e.dst
#     pos1, pos2 = crds[atomidx1], crds[atomidx2]
#     normaldir = SA[0.0, 0.0, 1.0]
#     if order > 1
#         ng1, ng2 = nbrs[atomidx1], nbrs[atomidx2]
#         # determine the plane for double bonds
#         if ng1 == 3
#             neighs = filter(x -> x != atomidx2, (neighbors(mol, atomidx1)))
#             @assert length(neighs) == 2
#             npos1, npos2 = crds[neighs[1]], crds[neighs[2]]
#             normaldir = cross(npos1, npos2)
#         elseif ng2 == 3
#             neighs = filter(x -> x != atomidx1, (neighbors(mol, atomidx2)))
#             @assert length(neighs) == 2
#             npos1, npos2 = crds[neighs[1]], crds[neighs[2]]
#             normaldir = cross(npos1, npos2)
#         end
#     end
#     sepdir = normalize(cross(normaldir, pos2 .- pos1))
#     dists = (bonddiameter * 2.5) .* collect(-0.5 * (order-1): 0.5 * (order-1))
#     for dist in dists
#         dvec = dist * sepdir
#         p1, p2 = pos1 + dvec, pos2 + dvec
#         if syms[atomidx1] == syms[atomidx2]
#             cyl = Cylinder(p1, p2, bonddiameter)
#             mesh!(f, cyl; color=col[atomidx1], kwargs...)
#         else
#             midpoint = 0.5 * (p1 + p2)
#             pm = Point(midpoint...)
#             cyl1 = Cylinder(p1, pm, bonddiameter)
#             cyl2 = Cylinder(p2, pm, bonddiameter)
#             mesh!(f, cyl1; color=col[atomidx1], kwargs...)
#             mesh!(f, cyl2; color=col[atomidx2], kwargs...)
#         end
#     end
#     return f
# end

