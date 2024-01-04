
from pdas.prom_utils import gen_sample_mesh

gen_sample_mesh(
    "random",
    "./full_mesh_decomp",
    "./sample_mesh_decomp",
    percpoints=0.2,
    randseed=2,
)

