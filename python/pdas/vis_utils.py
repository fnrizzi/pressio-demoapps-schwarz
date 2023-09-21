import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from pdas.data_utils import load_unified_helper
from pdas.prom_utils import load_pod_basis


# mpl.rc("font", family="serif", size="10")
mpl.rc("font", family="sans", size="10")
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
    draw_colorbar=True,
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

    meshlist, datalist = load_unified_helper(
        meshlist,
        datalist,
        meshdirs,
        datadirs,
        nvars,
        dataroot,
        merge_decomp=merge_decomp,
    )
    ndata = len(datalist)

    # prepping some lists
    if (plotlabels is None) or isinstance(plotlabels, str):
        plotlabels = [plotlabels] * ndata
    if not isinstance(plotskip, list):
        plotskip = [plotskip] * ndata
    assert len(plotskip) == ndata

    # set up time plot progression
    nt_min = np.infty
    for data_idx, data in enumerate(datalist):
        nt_min = min(nt_min, int(data.shape[-2] / plotskip[data_idx]))
    pause_time = plottime / nt_min

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
    for t in range(0, nt_min):

        for plotnum in range(ndata):
            ax[plotnum].cla()

            # plot monolithic or merged decomposed solution
            if True:
                cf = ax[plotnum].contourf(
                    meshlist[plotnum][:, :, 0],
                    meshlist[plotnum][:, :, 1],
                    datalist[plotnum][:, :, t * plotskip[plotnum], varplot],
                    levels=levels,
                    extend="both",
                )

            # plot non-combined decomposed solution
            else:
                raise ValueError("Decomposed solution plotting not implemented")

            ax[plotnum].set_title(plotlabels[plotnum], fontsize=18)
            ax[plotnum].tick_params(axis='both', which='major', labelsize=14)

        fig.supxlabel('x', fontsize=16)
        fig.supylabel('y', fontsize=16)

        if (t == 0) and draw_colorbar:
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
                cbar.ax.xaxis.set_label_position('top')
            cbar.set_ticks(ticks)
            if varlabel is not None:
                cbar.set_label(varlabel)

        plt.pause(pause_time)
        if (stopiter != -1) and (itercounter >= stopiter):
            breakpoint()

        if savefigs:
            plt.savefig(os.path.join(outdir, f'fig_{t}.png'))

        itercounter += 1

    plt.show()
    print("Finished")


def plot_lines(
    ylist,
    outdir,
    outname,
    linecolors,
    xlist=None,
    varplot=None,
    linestyles="-",
    xlabel=None,
    ylabel=None,
    legendlabels=None,
    legendloc="best",
    xbounds=[None,None],
    ybounds=[None,None],
    figdim=[6.4, 4.8],
):
    """
    Inputs:
        - xlist: a (list of) 1D array(s)
        - ylist: a (list of) 1D/2D arrays. If 2D, must provide varplot, and
            assumed that trailing dimension is variable dimension
    """

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    if not isinstance(ylist, list):
        ylist = [ylist]
    nlines = len(ylist)

    if legendlabels is not None:
        assert len(legendlabels) == nlines

    # must have unique colors for plots
    if not isinstance(linecolors, list):
        linecolors = [linecolors]
    assert len(linecolors) >= nlines

    # non-unique linestyles fine
    if not isinstance(linestyles, list):
        linestyles = [linestyles] * nlines
    assert len(linestyles) >= nlines

    # get 1D arrays
    for yidx, yarr in enumerate(ylist):
        assert yarr.ndim <= 2

        if yarr.ndim == 2:
            assert varplot is not None
            ylist[yidx] = yarr[:, varplot]

    # check dimensions
    nvals_list = [y.shape[0] for y in ylist]
    allsame = all([val == nvals_list[0] for val in nvals_list])

    # handling certain edge cases for xlist
    # WARNING: may be incorrect if same number of samples but over different range
    #   Better to just be explicit
    if xlist is None:
        assert allsame
        xlist = [np.arange(ylist[0].shape[0])] * nlines
    else:
        if not isinstance(xlist, list):
            assert allsame
            xlist = [xlist] * nlines
    assert len(xlist) == nlines

    # plot
    fig, ax = plt.subplots(1, 1, figsize=figdim)
    for line_idx in range(nlines):

        ax.plot(
            xlist[line_idx], ylist[line_idx],
            color=linecolors[line_idx], linestyle=linestyles[line_idx]
        )


    if legendlabels is not None:
        ax.legend(legendlabels, loc=legendloc)
    ax.set_xlim(xbounds)
    ax.set_ylim(ybounds)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()
    outfile = os.path.join(outdir, outname + ".png")
    print(f"Saving image to {outfile}")
    plt.savefig(outfile)


def plot_pod_res_energy(
    outdir,
    basis_dirs=None,
    svals=None,
    xlim=[None, None],
    plotcolors=["k", "r", "b"],
    linestyles=["-", "--", ":"],
    legend_labels=None,
    outsuff="",
):

    assert os.path.isdir(outdir)
    assert (basis_dirs is not None) != (svals is not None)

    if svals is None:
        if not isinstance(basis_dirs, list):
            basis_dirs = [basis_dirs]
        svals = []
        for basis_dir in basis_dirs:
            svals_in, = load_pod_basis(basis_dir, return_basis=False, return_svals=True)
            if isinstance(svals_in, list):
                svals += svals_in
            else:
                svals.append(svals_in.copy())
    else:
        if not isinstance(svals, list):
            assert isinstance(svals, np.ndarray)
            svals = [svals]

    nlines = len(svals)
    assert len(plotcolors) >= nlines
    assert len(linestyles) >= nlines
    if legend_labels is not None:
        assert len(legend_labels) == nlines

    fig, ax = plt.subplots(1, 1)
    for line_idx, sval_arr in enumerate(svals):

        # compute residual energy
        svals_sq = np.square(sval_arr)
        energy = 100* (1.0 - np.cumsum(svals_sq) / np.sum(svals_sq))

        nmodes = np.arange(1, energy.shape[0]+1)

        ax.semilogy(nmodes, energy, color=plotcolors[line_idx], linestyle=linestyles[line_idx])

    ax.set_xlim(xlim)
    ax.set_ylim([1e-2, 100])
    ax.set_xlabel("# POD modes")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_ylabel("Residual POD energy, %")
    if legend_labels is not None:
        ax.legend(legend_labels, loc="upper right", fontsize=12)
    ax.set_yticks(
        [1e-2, 1e-1, 1, 10, 100],
        ["0.01", "0.1", "1", "10", "100"],
    )

    plt.tight_layout()
    outfile = os.path.join(outdir, f"S{outsuff}.png")
    print(f"Saving image to {outfile}")
    plt.savefig(outfile)
