
from pdas.prom_utils import gen_pod_bases

gen_pod_bases(
    outdir="./trial_space",
    meshdir="../eigen_2d_swe_slip_wall_implicit/firstorder/",
    datadir="../eigen_2d_swe_slip_wall_implicit/firstorder/",
    nvars=3,
    dataroot="swe_slipWall2d_solution",
    center_method="init_cond",
    norm_method="one",
)
