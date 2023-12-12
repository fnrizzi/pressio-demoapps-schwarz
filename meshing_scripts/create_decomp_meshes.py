import os
import subprocess
from argparse import ArgumentParser, RawTextHelpFormatter
from math import floor

import numpy as np


def prep_dim(N, ndom, bounds):
    d = (bounds[1] - bounds[0]) / N
    N_dom = [int(N / ndom)] * ndom
    fill = N - int(N / ndom) * ndom
    for idx in range(fill):
        N_dom[idx] += 1

    return d, N_dom


def prep_dom_dim(ndom, N_dom, overlap, bounds, d):

    # cells that need to be distributed into overlap regions
    tot_overlap = overlap * (ndom - 1)
    added = floor(tot_overlap / ndom)
    extra = tot_overlap % ndom

    # cell counts for each domain after adding overlap
    added_bound = [[0, 0] for _ in range(ndom)]
    N_dom_overlap = [N for N in N_dom]
    for dom_idx in range(ndom):
        to_add = added
        if dom_idx >= (ndom - extra):
            to_add += 1
        N_dom_overlap[dom_idx] += to_add

        if dom_idx == 0:
            added_bound[dom_idx][0] = 0
            added_bound[dom_idx][1] = added
        else:
            added_bound[dom_idx][0] = overlap - added_bound[dom_idx-1][1]
            added_bound[dom_idx][1] = to_add - added_bound[dom_idx][0]

    # subdomain bounds
    bound = [[None, None] for _ in range(ndom)]
    for dom_idx in range(ndom):

        bound[dom_idx][0] = bounds[0] + (sum(N_dom[:dom_idx]) - added_bound[dom_idx][0]) * d
        bound[dom_idx][1] = bounds[0] + (sum(N_dom[:dom_idx+1]) + added_bound[dom_idx][1]) * d

    return N_dom_overlap, bound


def get_linear_index(numdoms_list, i, j, k):
    return i + numdoms_list[0] * j + (numdoms_list[0] + numdoms_list[1]) * k


def main(
    numcells_list,
    bounds_list,
    stencilsize,
    numdoms_list,
    overlap,
    outdir,
    mesh_script,
    stdout=False,
):

    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    assert os.path.isfile(mesh_script)

    ndim = len(numcells_list)

    # input size checks
    assert ndim in [1, 2, 3]
    assert stencilsize in [3, 5, 7]
    assert len(bounds_list) == 2 * ndim
    assert len(numdoms_list) == ndim

    numdoms_list += [1] * (3 - ndim) # pad with ones for loop mechanics
    numdoms = np.prod(numdoms_list)

    # spatial input checks
    for dim in range(ndim):
        assert numcells_list[dim] > 0
        assert bounds_list[2*dim+1] > bounds_list[2*dim]
        assert numdoms_list[ndim] > 0
    assert overlap >= 0
    # TODO: permit periodic domains

    # domain dimensions
    dx_list = [None for _ in range(ndim)]
    ncells_dom_list = [None for _ in range(ndim)]
    for dim in range(ndim):
        dx_list[dim], ncells_dom_list[dim] = prep_dim(
            numcells_list[dim],
            numdoms_list[dim],
            bounds_list[2*dim:2*dim+2]
        )

    # subdomain dimensions
    dom_grid_idxs = [None for _ in range(numdoms)]
    for dom_idx in range(numdoms):
        # gridded indices
        dom_grid_idxs[dom_idx] = [
            dom_idx % numdoms_list[0],
            int(dom_idx / numdoms_list[0]),
            int(dom_idx / (numdoms_list[0] * numdoms_list[1])),
        ]

    numcells_sub_list = [[1 for _ in range(numdoms_list[dim])] for dim in range(3)]
    bounds_sub_list = [None for _ in range(ndim)]
    for dim in range(ndim):
        numcells_sub_list[dim], bounds_sub_list[dim] = prep_dom_dim(
            numdoms_list[dim],
            ncells_dom_list[dim],
            overlap,
            bounds_list[2*dim:2*dim+2],
            dx_list[dim],
        )

    # linear subdomain index
    for dom_idx in range(numdoms):

        if stdout:
            print(f"Domain {dom_idx}")
            print("Cells: " + " x ".join(map(str, [numcells_sub_list[dim][dom_grid_idxs[dom_idx][dim]] for dim in range(ndim)])))
            for dim in range(ndim):
                print(["x", "y", "z"][dim] + f"-bounds: {bounds_sub_list[dim][dom_grid_idxs[dom_idx][dim]]}")

        # subdomain mesh subdirectory
        outdir_sub = os.path.join(outdir, "domain_" + str(dom_idx))
        if not os.path.isdir(outdir_sub):
            os.makedirs(outdir_sub)
        if stdout:
            print("Output directory: " + outdir_sub)

        # command line execution
        arg_tuple = ("python3", mesh_script,)
        arg_tuple += ("-n",) + tuple([str(numcells_sub_list[dim][dom_grid_idxs[dom_idx][dim]]) for dim in range(ndim)])
        arg_tuple += ("--outDir", outdir_sub,)
        arg_tuple += ("--stencilsize", str(stencilsize),)
        blep = [[str(bounds_sub_list[dim][dom_grid_idxs[dom_idx][dim]][bound]) for bound in range(2)] for dim in range(ndim)]
        arg_tuple += ("--bounds",) + tuple(sum([[str(bounds_sub_list[dim][dom_grid_idxs[dom_idx][dim]][bound]) for bound in range(2)] for dim in range(ndim)], []))
        popen  = subprocess.Popen(arg_tuple, stdout=subprocess.PIPE); popen.wait()

        # load generated connectivity
        connect = np.loadtxt(os.path.join(outdir_sub, "connectivity.dat"), dtype=np.int32)[:, 1:]

        i = dom_grid_idxs[dom_idx][0]
        j = dom_grid_idxs[dom_idx][1]
        k = dom_grid_idxs[dom_idx][2]

        # write neighbor subdomain stencil connectivity
        with open(os.path.join(outdir_sub, "connectivity_neighbor.dat"), "w") as f:
            ncells = np.prod([numcells_sub_list[dim][curr_idx] for dim, curr_idx in zip(range(3), [i, j, k])])
            for cell_idx in range(ncells):
                f.write(f"{cell_idx:8d}")

                stencil_gids = connect[cell_idx, :]

                x_idx = cell_idx % numcells_sub_list[0][i]
                if ndim > 1:
                    y_idx = int(cell_idx / numcells_sub_list[0][i])
                if ndim == 3:
                    z_idx = int(cell_idx / (numcells_sub_list[0][i] * numcells_sub_list[1][j]))


                for stencil_idx in range(int((stencilsize - 1) / 2)):
                    for axis_idx in range(ndim * 2):

                        connect_idx = stencil_idx * ndim * 2 + axis_idx
                        stencil_gid = stencil_gids[connect_idx]

                        neigh_gid = -1

                        if stencil_gid == -1:
                            # left subdomain
                            if (axis_idx == 0) and (i != 0):
                                numcells_list_neigh = [numcells_sub_list[dim][neigh_idx] for dim, neigh_idx in zip(range(3), [i-1, j, k])]
                                dist = x_idx
                                neigh_gid = (numcells_list_neigh[0] * (y_idx + 1)) - overlap - stencil_idx + dist - 1
                            # right subdomain (1D)
                            if ndim == 1:
                                if (axis_idx == 1) and (i != numdoms_list[0] - 1):
                                    dist = numcells_sub_list[0][i] - x_idx - 1
                                    neigh_gid = overlap + stencil_idx - dist

                            if ndim > 1:
                                # front boundary
                                if (axis_idx == 1) and (j != numdoms_list[1] - 1):
                                    numcells_list_neigh = [numcells_sub_list[dim][neigh_idx] for dim, neigh_idx in zip(range(3), [i, j+1, k])]
                                    dist = numcells_sub_list[1][j] - y_idx - 1
                                    neigh_gid = (overlap + stencil_idx - dist) * numcells_list_neigh[0] + x_idx

                                # right boundary (2D)
                                if (axis_idx == 2) and (i != numdoms_list[0] - 1):
                                    numcells_list_neigh = [numcells_sub_list[dim][neigh_idx] for dim, neigh_idx in zip(range(3), [i+1, j, k])]
                                    dist = numcells_sub_list[0][i] - x_idx - 1
                                    neigh_gid = (numcells_list_neigh[0] * y_idx) + overlap + stencil_idx - dist

                                # back boundary
                                if (axis_idx == 3) and (j != 0):
                                    numcells_list_neigh = [numcells_sub_list[dim][neigh_idx] for dim, neigh_idx in zip(range(3), [i, j-1, k])]
                                    dist = y_idx
                                    neigh_gid = (numcells_list_neigh[1] - 1 - overlap - stencil_idx + dist) * numcells_list_neigh[0] + x_idx

                            if ndim == 3:
                                raise ValueError("3D not completed")

                        f.write(f" {neigh_gid:8d}")
                f.write("\n")

    with open(os.path.join(outdir, "info_domain.dat"), "w") as f:
        f.write("dim %8d\n" % ndim)
        f.write("ndomX %8d\n" % numdoms_list[0])
        if ndim > 1:
            f.write("ndomY %8d\n" % numdoms_list[1])
            if ndim == 3:
                f.write("ndomZ %8d\n" % numdoms_list[2])

        f.write("overlap %8d\n" % overlap)

if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)

    # location of create_full_mesh.py
    parser.add_argument(
        "--meshScript", "--meshscript", "--mesh_script",
        dest="mesh_script",
        help="Full path to create_full_mesh.py supplied by pressio-demoapps",
    )

    # these are mostly copied from create_full_mesh.py for consistency sake
    parser.add_argument(
        "--outDir", "--outdir", "--out_dir",
        dest="outdir",
        help="Full path to output directory where all the mesh files will be generated.",
    )

    parser.add_argument(
        "-n", "--numCells", "--numcells", "--num_cells",
        nargs="*",
        type=int,
        dest="numcells",
        help="TOTAL number of cells to use along each axis. This determines the dimensionality.\n"+
            "If you pass one value ,   I assume a 1d domain.\n"+
            "If you pass two values,   I assume a 2d domain.\n"+
            "If you pass three values, I assume a 3d domain.",
    )

    parser.add_argument(
        "-b", "--bounds",
        nargs="*",
        type=float,
        dest="bounds",
        help="Domain physical bounds along each axis. Must be a list of pairs.\n"+
            "First, you pass bounds for x axis, then, if needed, for y, and z.\n"+
            "For example: \n"+
            "  --bounds 0.0 1.0           implies that x is in (0,1)\n"+
            "  --bounds 0.0 1.0 -1.0 1.0: implies x in (0,1) and y in (-1,1).\n"+
            "NOTE: the number of pairs passed to --bounds must be consistent \n"+
            "      with how many args you specify for --numCells. ",
    )

    parser.add_argument(
        "-s", "--stencilSize", "--stencilsize", "--stencil_size",
        type=int, dest="stencilsize", default=3,
        choices=[3,5,7],
        help="Stencil size to use for assembling the connectivity",
    )

    # these are specific to Schwarz
    parser.add_argument(
        "--numdoms", "--numDoms", "--num_doms",
        nargs="*",
        type=int,
        dest="numdoms",
        help="Number of subdomain divisions in each direction.\n"+
            "If you pass one value ,   I assume a 1d domain.\n"+
            "If you pass two values,   I assume a 2d domain.\n"+
            "If you pass three values, I assume a 3d domain.",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        dest="overlap",
        help="Number of cells in subdomain overlap region\n"+
            "If you pass > 0, will use overlapping Dirichlet-Dirichlet coupling\n"+
            "If you pass > 0, will use non-overlapping Dirichlet-Neumann coupling",
    )

    # these should NOT change for now
    # TODO: allow for handling periodic BCs
    periodic = False
    ordering_type = "natural_row"

    argobj = parser.parse_args()

    # NOTE: input checks are peformed in main()

    main(
        argobj.numcells,
        argobj.bounds,
        argobj.stencilsize,
        argobj.numdoms,
        argobj.overlap,
        argobj.outdir,
        argobj.mesh_script,
        stdout=True,
    )
