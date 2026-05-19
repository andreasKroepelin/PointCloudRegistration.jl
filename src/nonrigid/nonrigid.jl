abstract type NonRigidRegistration end

function (nr::NonRigidRegistration)(pc::PointCloud)
    error("non-rigid registration of type $(typeof(nr)) does not provide " *
        "displacements for arbitrary points")
end

function correspondences end

function nonrigid_sinkhorn end
function nonrigid_divfree end
