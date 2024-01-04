
from pdas.vis_utils import plot_contours

# ----- START USER INPUTS -----

# 0: density
# 1: x-momentum
# 2: y-momentum
# 3: total energy
varplot = 0

# ----- END USER INPUTS -----

if varplot == 0:
    varlabel = r"Density"
    nlevels = 15
    skiplevels = 1
    contourbounds = [0.1, 1.5]
elif varplot == 1:
    varlabel = r"X-momentum"
    nlevels = 21
    skiplevels = 2
    contourbounds = [-0.5, 0.5]
elif varplot == 2:
    varlabel = r"Y-momentum"
    nlevels = 21
    skiplevels = 2
    contourbounds = [-0.5, 0.5]
elif varplot == 3:
    varlabel = r"Energy"
    nlevels = 15
    skiplevels = 2
    contourbounds = [0.25, 3.75]

plot_contours(
    varplot,
    meshdirs="./",
    datadirs="./",
    nvars=4,
    dataroot="riemann2d_solution",
    nlevels=nlevels,
    skiplevels=skiplevels,
    contourbounds=contourbounds,
    plotskip=1,
    varlabel=varlabel,
)
