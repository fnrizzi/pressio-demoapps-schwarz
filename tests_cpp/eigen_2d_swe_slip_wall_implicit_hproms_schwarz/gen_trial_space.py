
import numpy as np

from pdas.prom_utils import gen_pod_bases

data = np.loadtxt("../../../eigen_2d_swe_slip_wall_implicit/firstorder/solution_full_gold.txt")
data = np.reshape(data, (50, 50, 3, -1), order="C")
data = np.transpose(data, (1, 0, 3, 2))

gen_pod_bases(
    outdir="./trial_space",
    meshdir="../../../eigen_2d_swe_slip_wall_implicit/firstorder/",
    datalist=[data],
    nvars=3,
    dataroot="swe_slipWall2d_solution",
    pod_decomp=True,
    meshdir_decomp="./full_mesh",
    center_method="init_cond",
    norm_method="one",
)

