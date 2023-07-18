import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from pdas.data_utils import load_meshes, load_field_data


mpl.rc("font", family="serif", size="10")
mpl.rc("axes", labelsize="x-large")
mpl.rc("figure", facecolor="w")
mpl.rc("text", usetex=False)


def plot_contours(
    varplot,
    meshlist=None,
    datalist=None,
    meshdirs=None,
    datadirs=None,
    nvars=None,
    dataroot=None,
    merge_decomp=True,
    savefigs=False,
    outdir=None,
    plotlabels=None,
    nlevels=20,
    skiplevels=1,
    contourbounds=[None,None],
    plottime=10,
    plotskip=1,
    stopiter=-1,
    varlabel=None,
    plotbounds=True,
    bound_colors=None,
    figdim_base=[6.4, 4.8],
    vertical=True,
):

    # TODO: check dimension, slice if 3D

    # some input checking
    if stopiter != -1:
        assert stopiter >= 0
    if savefigs:
        assert outdir is not None
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
    if meshlist is not None:
        if isinstance(meshlist, np.ndarray):
            meshlist = [meshlist]
        elif not isinstance(meshlist, list):
            raise ValueError("meshlist expected to be a list or numpy array")
    if datalist is not None:
        if isinstance(datalist, np.ndarray):
            datalist = [datalist]
        elif not isinstance(datalist, list):
            raise ValueError("datalist expected to be a list or numpy array")
    if isinstance(meshdirs, str):
        meshdirs = [meshdirs]
    if isinstance(datadirs, str):
        datadirs = [datadirs]

    # load meshes, if not provided
    if meshlist is None:
        assert meshdirs is not None
        nmesh = len(meshdirs)
        meshlist     = [None for _ in range(nmesh)]
        meshlist_sub = [None for _ in range(nmesh)]
        for mesh_idx, meshdir in enumerate(meshdirs):
            meshlist[mesh_idx], meshlist_sub[mesh_idx] = load_meshes(meshdir, merge_decomp=merge_decomp)
    else:
        nmesh = len(meshlist)

    # load data, if not provided
    if datalist is None:
        assert meshdirs is not None
        assert datadirs is not None
        assert dataroot is not None
        assert nvars is not None
        ndata = len(datadirs)
        assert nmesh == ndata
        assert len(meshdirs) == ndata # in case meshlist was provided

        datalist = [None for _ in range(ndata)]
        datalist_sub = [None for _ in range(ndata)]
        for data_idx, datadir in enumerate(datadirs):
            # is monolithic
            if meshlist_sub[data_idx] is None:
                coords_in = meshlist[data_idx]
            else:
                coords_in = meshlist_sub[data_idx]
            datalist[data_idx], datalist_sub[data_idx] = load_field_data(
                datadir,
                dataroot,
                nvars,
                coords=coords_in,
                meshdir=meshdirs[data_idx],
                merge_decomp=merge_decomp,
            )
    else:
        ndata = len(datalist)
        assert nmesh == ndata

    # prepping some lists
    if (plotlabels is None) or isinstance(plotlabels, str):
        plotlabels = [plotlabels] * ndata

    # set up time plot progression
    nt_min = np.infty
    for data in datalist:
        nt_min = min(nt_min, data.shape[-2])
    pause_time = plottime / (nt_min / plotskip)

    # set up axes
    if vertical:
        fig, ax = plt.subplots(nrows=ndata, ncols=1)
        fig.set_figwidth(figdim_base[0])
        fig.set_figheight(figdim_base[1] * ndata)
    else:
        fig, ax = plt.subplots(nrows=1, ncols=ndata)
        fig.set_figwidth(figdim_base[0] * ndata)
        fig.set_figheight(figdim_base[1])
    if not isinstance(ax, np.ndarray):
         ax = np.array([ax])

    # set up contour parameters
    # assume first plot provides reasonable bounds, TODO: could improve
    if any([bound is None for bound in contourbounds]):
        contourbounds[0] = np.amin(datalist[0][:, :, :, varplot])
        contourbounds[1] = np.amax(datalist[0][:, :, :, varplot])
    levels = np.linspace(contourbounds[0], contourbounds[1], nlevels)
    ticks = levels[::skiplevels]

    itercounter = 0
    for t in range(0, nt_min, plotskip):

        for plotnum in range(ndata):
            ax[plotnum].cla()

            # plot monolithic or merged decomposed solution
            if True:
                cf = ax[plotnum].contourf(
                    meshlist[plotnum][:, :, 0],
                    meshlist[plotnum][:, :, 1],
                    datalist[plotnum][:, :, t, varplot],
                    levels=levels,
                    extend="both",
                )

            # plot non-combined decomposed solution
            else:
                raise ValueError("Decomposed solution plotting not implemented")

            ax[plotnum].set_title(plotlabels[plotnum], fontsize=18)
            ax[plotnum].tick_params(axis='both', which='major', labelsize=14)

        fig.supxlabel('x (m)', fontsize=16)
        fig.supylabel('y (m)', fontsize=16)

        if t == 0:
            plt.tight_layout()
            if vertical:
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.1, 0.025, 0.8])
                cbar = fig.colorbar(cf, cax=cbar_ax, orientation="vertical")
            else:
                # fig.subplots_adjust(bottom=0.2)
                # cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.025])
                fig.subplots_adjust(top=0.8)
                cbar_ax = fig.add_axes([0.1, 0.9, 0.8, 0.025])
                cbar = fig.colorbar(cf, cax=cbar_ax, orientation="horizontal")
            cbar.set_ticks(ticks)
            if varlabel is not None:
                cbar.set_label(varlabel)

        plt.pause(pause_time)
        if (stopiter != -1) and (itercounter >= stopiter):
            breakpoint()

        if savefigs:
            plotiter = int(t / plotskip)
            plt.savefig(os.path.join(outdir, f'fig_{plotiter}.png'))

        itercounter += 1

    plt.show()
    print("Finished")
