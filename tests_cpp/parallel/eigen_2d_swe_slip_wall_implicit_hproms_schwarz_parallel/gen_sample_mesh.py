import os
from argparse import ArgumentParser

from pdas.prom_utils import gen_sample_mesh

parser = ArgumentParser()
parser.add_argument("--outdir", dest="outdir")
args = parser.parse_args()

gen_sample_mesh(
    "random",
    os.path.join(args.outdir, "full_mesh_decomp"),
    os.path.join(args.outdir, "sample_mesh_decomp"),
    percpoints=0.2,
    randseed=2,
    samp_bounds=True,
)

