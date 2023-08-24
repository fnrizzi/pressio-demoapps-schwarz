
from pdas.data_utils import load_field_data
from pdas.prom_utils import load_reduced_data
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

fom_data, _ = load_field_data(
    "../../../eigen_2d_swe_slip_wall_implicit/firstorder/",
    "swe_slipWall2d_solution",
    3,
    meshdir="../../../eigen_2d_swe_slip_wall_implicit/firstorder/",
)

rom_data = load_reduced_data(
    "./",
    "swe_slipWall2d_solution",
    3,
    "./",
    "../../trial_space/",
    "center",
    "basis",
    25,
)

plot_contours(
    varplot,
    meshdirs=["../../../eigen_2d_swe_slip_wall_implicit/firstorder/", "./",],
    datalist=[fom_data, rom_data],
    nvars=3,
    dataroot="swe_slipWall2d_solution",
    plotlabels=["Monolithic", "Schwarz, 2x2"],
    nlevels=nlevels,
    skiplevels=skiplevels,
    contourbounds=contourbounds,
    plotskip=2,
    varlabel=varlabel,
    figdim_base=[8, 9],
    vertical=False,
)
