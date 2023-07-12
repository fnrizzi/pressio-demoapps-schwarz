import os
import subprocess
from argparse import ArgumentParser, RawTextHelpFormatter


def prep_dim(N, ndom, bounds):
    d = (bounds[1] - bounds[0]) / N
    N_dom = [int(N / ndom)] * ndom
    fill = N - int(N / ndom) * ndom
    for idx in range(fill):
        N_dom[idx] += 1

    return d, N_dom

def prep_dom_dim(dom_idx, ndom, N_dom, offset, bounds, d):

    # cells
    n = N_dom[dom_idx]
    if ndom > 1:
        if (dom_idx == 0) or (dom_idx == (ndom - 1)):
            n += offset
        else:
            n += 2 * offset

    # bounds
    bound = [None, None]
    if dom_idx == 0:
        bound[0] = bounds[0]
    else:
        bound[0] = bounds[0] + (sum(N_dom[:dom_idx]) - offset) * d
    if (dom_idx == ndom - 1):
        bound[1] = bounds[1]
    else:
        bound[1] = bounds[0] + (sum(N_dom[:dom_idx+1]) + offset) * d

    return n, bound

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
    numdoms = 1
    for dim in range(ndim):
        numdoms *= numdoms_list[dim]

    # spatial input checks
    for dim in range(ndim):
        assert numcells_list[dim] > 0
        assert bounds_list[2*dim+1] > bounds_list[2*dim]
        assert numdoms_list[ndim] > 0
    assert overlap >= 0
    assert int(overlap/2) == (overlap / 2), "Overlap must be even for now (FIX)"
    # TODO: permit periodic domains

    offset = int(overlap / 2)

    # domain dimensions
    dx_list = [None for _ in range(ndim)]
    ncells_dom_list = [None for _ in range(ndim)]
    for dim in range(ndim):
        dx_list[dim], ncells_dom_list[dim] = prep_dim(
            numcells_list[dim],
            numdoms_list[dim],
            bounds_list[2*dim:2*dim+2]
        )

    # linear subdomain index
    for dom_idx in range(numdoms):

        if stdout:
            print("Domain " + str(dom_idx))

        # gridded indices
        dom_grid_idxs = [
            dom_idx % numdoms_list[0],
            int(dom_idx / numdoms_list[0]),
            int(dom_idx / (numdoms_list[0] * numdoms_list[1])),
        ]

        numcells_sub_list = [None for _ in range(ndim)]
        bounds_sub_list = [None for _ in range(2*ndim)]

        # subdomain dimensions
        for dim in range(ndim):
            numcells_sub_list[dim], bounds_sub_list[2*dim:2*dim+1] = prep_dom_dim(
                dom_grid_idxs[dim],
                numdoms_list[dim],
                ncells_dom_list[dim],
                offset,
                bounds_list[2*dim:2*dim+2],
                dx_list[dim],
            )
        if stdout:
            print("Cells: " + " x ".join(map(str, numcells_sub_list)))
            for dim in range(ndim):
                print(["x", "y", "z"][dim] + f"-bounds: ({bounds_sub_list[2*dim]}, {bounds_sub_list[2*dim+1]})")

        # subdomain mesh subdirectory
        outdir_sub = os.path.join(outdir, "domain_" + str(dom_idx))
        if not os.path.isdir(outdir_sub):
            os.makedirs(outdir_sub)
        if stdout:
            print("Output directory: " + outdir_sub)

        # command line execution
        arg_tuple = ("python3", mesh_script,)
        arg_tuple += ("-n",) + tuple([str(val) for val in numcells_sub_list])
        arg_tuple += ("--outDir", outdir_sub,)
        arg_tuple += ("--stencilsize", str(stencilsize),)
        arg_tuple += ("--bounds",) + tuple([str(val) for val in bounds_sub_list[:2*ndim]])
        popen  = subprocess.Popen(arg_tuple, stdout=subprocess.PIPE); popen.wait()

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
