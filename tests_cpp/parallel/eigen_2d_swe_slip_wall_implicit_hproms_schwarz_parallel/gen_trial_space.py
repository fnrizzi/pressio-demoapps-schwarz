import os
import re
from argparse import ArgumentParser

import numpy as np

from pdas.prom_utils import gen_pod_bases

parser = ArgumentParser()
parser.add_argument("--outdir", dest="outdir")
args = parser.parse_args()

order = os.path.basename(os.path.normpath(args.outdir))
order = re.sub('_tp', '', order)
order = re.sub('_omp', '', order)

data = np.loadtxt(f"../../eigen_2d_swe_slip_wall_implicit_large/{order}/solution_full_gold.txt")
data = np.reshape(data, (100, 90, 3, -1), order="C")
data = np.transpose(data, (1, 0, 3, 2))

gen_pod_bases(
    outdir=os.path.join(args.outdir, "trial_space"),
    meshdir=os.path.join(args.outdir, "full_mesh_mono"),
    datalist=[data],
    nvars=3,
    dataroot="swe_slipWall2d_solution",
    pod_decomp=True,
    meshdir_decomp=os.path.join(args.outdir, "full_mesh_decomp"),
    center_method="init_cond",
    norm_method="one",
)

