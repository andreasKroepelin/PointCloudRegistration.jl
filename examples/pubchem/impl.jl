function cached_get_cid(name)
    cachefile = "cache-cid-$name"
    if !isfile(cachefile)
        @info "missing cache file for $name, calling PubChem API"
        cid = get_cid(; name)
        write(cachefile, cid)
    end
    return read(cachefile, Int64)
end

function load_pubchem_data(cids)
    all_cids = Int64[]
    for file in readdir()
        startswith(file, "cache-cid-") || continue
        push!(all_cids, read(file, Int64))
    end
    append!(all_cids, values(cids))
    unique!(all_cids)
    needs_download = true
    if isfile("cache-pubchem")
        sys = load_pubchem_json("cache-pubchem")
        if all(cid -> find_compound(sys, cid) !== nothing, all_cids)
            needs_download = false
        end
    end
    if needs_download
        @info "Downloading PubChem data for $(length(all_cids)) compounds."
        pubchem_res = get_for_cids(all_cids; output = "JSON", record_type = "3d")
        write("cache-pubchem", pubchem_res)
        sys = load_pubchem_json("cache-pubchem")
    end
    return map(cid -> find_compound(sys, cid), cids)
end

function find_compound(sys, cid)
    i = findfirst(m -> m.name == "CID $cid", molecules(sys))
    isnothing(i) && return nothing
    return molecules(sys)[i]
end

get_atomtypes(compound) = compound.atoms.element |> Vector{Int}

function get_coords(compound)
    conformer = compound.coords[1].conformers[1]
    arr = stack(["x", "y", "z"]) do ax
        Vector{Float64}(conformer[ax])
    end
    return copy(transpose(arr))
end

function get_bonds(compound)
    bs = NTuple{2, Int64}[]
    for b in bonds(compound)
        (; a1, a2) = b
        i = findfirst(at -> at.idx == a1, atoms(compound))
        j = findfirst(at -> at.idx == a2, atoms(compound))
        push!(bs, (i, j))
    end
    return bs
end

function atomtypes2colors(atomtypes)
    map(atomtypes) do at
        if at == Elements.H
            :white
        elseif at == Elements.C
            :black
        elseif at == Elements.N
            :blue
        elseif at == Elements.O
            :red
        else
            :gray
        end
    end
end
