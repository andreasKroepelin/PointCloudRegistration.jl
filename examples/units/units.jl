using Revise
using Unitful
import Unitful: Å
using PointCloudRegistration
import PointCloudRegistration as PCReg
using DelimitedFiles

coords = rand(2, 10) .* 1u"m"
pc = PointCloud(coords)

register_rmsd(pc, pc)

register_gmc(pc, pc; scale = DownTo(1u"cm"))

prepd_pc = prepare_target_kc(pc; scale = DownTo(1u"cm"))
register_kc(pc, prepd_pc)

pc_1ake = PointCloud(readdlm("../../assets/ake/1ake.csv", ',') .* 1Å)
pc_4ake = PointCloud(readdlm("../../assets/ake/4ake.csv", ',') .* 1Å)

register_rmsd(pc_1ake, pc_4ake)
register_gmc(pc_1ake, pc_4ake; scale = DownTo(1Å))
prepd_4ake = prepare_target_kc(pc_4ake; scale = DownTo(1Å));
T = register_kc(pc_1ake, prepd_4ake; restarts = RandomRestarts(100))

T_pc_1ake = T(pc_1ake)
cpd = register_cpd(
    T_pc_1ake,
    pc_4ake;
    scale = 1Å,
    outlier_proportion = 0.01,
    regularizer_strength = 0.1Å^(-2),
    regularizer_lengthscale = 5Å,
)
