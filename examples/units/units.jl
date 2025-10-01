using Revise
using Unitful
using PointCloudRegistration
import PointCloudRegistration as PCReg

coords = rand(2, 10) .* 1u"m"
pc = PointCloud(coords)

PCReg.annealing_plan(pc, DownTo(1u"cm"))
