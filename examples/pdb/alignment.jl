function aligned_atoms(
    el1,
    el2,
    residue_selectors::Function...;
    scoremodel::AbstractScoreModel = AffineGapScoreModel(
        BLOSUM62,
        gap_open = -10,
        gap_extend = -1,
    ),
    aligntype::BioAlignments.AbstractAlignment = LocalAlignment(),
    alignatoms::Function = calphaselector,
)
    res1 = collectresidues(el1, residue_selectors...)
    res2 = collectresidues(el2, residue_selectors...)
    # Shortcut if the sequences are the same
    if LongAA(res1; gaps = false) == LongAA(res2; gaps = false)
        inds1 = collect(1:length(res1))
        inds2 = collect(1:length(res2))
    else
        alres = pairalign(res1, res2; scoremodel, aligntype)
        al = alignment(alres)
        inds1, inds2 = Int[], Int[]
        # Offset residue counter based on start of aligned region
        first_anchor = first(al.a.aln.anchors)
        counter1 = first_anchor.seqpos
        counter2 = first_anchor.refpos
        # Obtain indices of residues used in alignment
        for (v1, v2) in al
            if v1 != AA_Gap
                counter1 += 1
            end
            if v2 != AA_Gap
                counter2 += 1
            end
            if v1 != AA_Gap && v2 != AA_Gap
                push!(inds1, counter1)
                push!(inds2, counter2)
            end
        end
    end
    @info "Found sequence alignment between $(length(inds1)) residues (out of $(length(res1)) / $(length(res2)))"
    atoms1, atoms2 = AbstractAtom[], AbstractAtom[]
    inds1_used, inds2_used = Int[], Int[]
    for (i1, i2) in zip(inds1, inds2)
        sel_ats1 = collectatoms(res1[i1], alignatoms)
        sel_ats2 = collectatoms(res2[i2], alignatoms)
        # Ensure `atoms1` and `atoms2` have the same length, ignore residues
        # where the number of atoms differ
        if length(sel_ats1) == length(sel_ats2)
            append!(atoms1, sel_ats1)
            append!(atoms2, sel_ats2)
            push!(inds1_used, i1)
            push!(inds2_used, i2)
        end
    end
    if length(atoms1) == 0
        throw(ArgumentError("No atoms found to superimpose"))
    end

    (PointCloud ∘ coordarray).((atoms1, atoms2))
end
