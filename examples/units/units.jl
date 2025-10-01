using Revise
using Unitful
using PointCloudRegistration
import PointCloudRegistration as PCReg
using DelimitedFiles

coords = rand(2, 10) .* 1u"m"
pc = PointCloud(coords)

register_rmsd(pc, pc)

register_gmc(pc, pc; scale = DownTo(1u"cm"))

prepd_pc = prepare_target_kc(pc; scale = DownTo(1u"cm"))
register_kc(pc, prepd_pc)

pc_1ake = PointCloud(readdlm("../../assets/ake/1ake.csv", ',') .* 1u"angstrom")
pc_4ake = PointCloud(readdlm("../../assets/ake/4ake.csv", ',') .* 1u"angstrom")

register_rmsd(pc_1ake, pc_4ake)
register_gmc(pc_1ake, pc_4ake; scale = DownTo(1u"angstrom"))
prepd_4ake = prepare_target_kc(pc_4ake; scale = DownTo(1u"angstrom"));
T = register_kc(pc_1ake, prepd_4ake; restarts = RandomRestarts(100))
T(pc_1ake)
