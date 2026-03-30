abstract type NonRigidRegistration end

function displacements end

function apply_displacements(pc::PointCloud{N}, displacements::VecOfSVec{N}) where {N}
    PointCloud(pc.points .+ displacements, pc.weights)
end

function (nr::NonRigidRegistration)(pc::PointCloud)
    error("non-rigid registration of type $(typeof(nr)) does not provide " *
        "displacements for arbitrary points")
end

function correspondences end

function nonrigid_sinkhorn end
function nonrigid_divfree end

