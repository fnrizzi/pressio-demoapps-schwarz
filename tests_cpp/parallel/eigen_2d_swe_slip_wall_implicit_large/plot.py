
from pdas.vis_utils import plot_contours

# ----- START USER INPUTS -----

# 0: height
# 1: x-momentum
# 2: y-momentum
varplot = 0

# ----- END USER INPUTS -----

if varplot == 0:
    varlabel = r"Height"
    nlevels = 25
    skiplevels = 2
    contourbounds = [1.0, 1.024]
elif varplot == 1:
    varlabel = r"X-momentum"
    nlevels = 21
    skiplevels = 2
    contourbounds = [-0.05, 0.05]
elif varplot == 2:
    varlabel = r"Y-momentum"
    nlevels = 21
    skiplevels = 2
    contourbounds = [-0.05, 0.05]

plot_contours(
    varplot,
    meshdirs="./",
    datadirs="./",
    nvars=3,
    dataroot="swe_slipWall2d_solution",
    nlevels=nlevels,
    skiplevels=skiplevels,
    contourbounds=contourbounds,
    plotskip=2,
    varlabel=varlabel,
)
