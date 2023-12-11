import os

from pdas.vis_utils import plot_contours

# ----- START USER INPUTS -----

# 0: density
# 1: x-momentum
# 2: y-momentum
# 3: total energy
varplot = 0

# ----- END USER INPUTS -----

exe_dir = os.path.dirname(os.path.realpath(__file__))
order = os.path.basename(os.path.normpath(exe_dir))

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

# TODO: modify monolithic directory to correct stencil order
plot_contours(
    varplot,
    meshdirs=[f"../../eigen_2d_euler_riemann_implicit/{order}", "./mesh"],
    datadirs=[f"../../eigen_2d_euler_riemann_implicit/{order}", "./"],
    nvars=4,
    dataroot="riemann2d_solution",
    plotlabels=["Monolithic", "Schwarz, 2x2"],
    nlevels=nlevels,
    skiplevels=skiplevels,
    contourbounds=contourbounds,
    plotskip=2,
    varlabel=varlabel,
    plotbounds=True,
    bound_colors=["b", "r", "m", "c"],
    figdim_base=[8, 9],
    vertical=False,
)
