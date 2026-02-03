abstract type NonRigidRegistration end

function displacements end

function apply_displacements(pc::PointCloud{N}, displacements::VecOfSVec{N}) where {N}
    PointCloud(pc.points .+ displacements, pc.weights)
end

function correspondences end

function register_sinkhorn end

