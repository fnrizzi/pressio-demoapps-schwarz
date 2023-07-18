import os

import numpy as np


def get_nested_decomp_dims(data_list):
    # NOTE: ALL data is assumed to reside in a list-of-lists-of-lists
    # Reflects a "3D" domain decomposition, even if y- or z-direction is 1 subdomain

    ndom_list = [1 for _ in range(3)]

    ndom_list[0] = len(data_list)

    ndom_list[1] = len(data_list[0])
    assert all([len(data) == ndom_list[1] for data in data_list])

    ndom_list[2] = len(data_list[0][0])
    for dimx in range(ndom_list[0]):
        assert all([len(data) == ndom_list[2] for data in data_list[dimx]])

    return ndom_list


def get_full_dims_from_decomp(data_list, overlap, ndom_list=None, is_ts=True):

    offset = round(overlap / 2)
    if ndom_list is None:
        ndom_list = get_nested_decomp_dims(data_list)

    # coordinate data should have ndim + 1 (spatial dim) dimensions
    # state data should have ndim + 2 (time, variable) dimensions
    ndim = len(data_list[0][0][0].shape) - 1
    if is_ts:
        ndim -= 1

    # combining grid dimensions, if requested
    cells_list = [np.zeros(tuple([ndom for ndom in ndom_list])) for _ in range(3)]
    cells_total_list = [0 for _ in range(3)]

    # individual domain shapes
    for i in range(ndom_list[0]):
        for j in range(ndom_list[1]):
            for k in range(ndom_list[2]):
                cells = list(data_list[i][j][k].shape)
                for dim in range(ndim):
                    cells_list[dim][i, j, k] = cells[dim]

    # x-direction
    for i in range(ndom_list[0]):
        cells_total_list[0] += data_list[i][0][0].shape[0]
        if i != 0:
            cells_total_list[0] -= offset
        if i != (ndom_list[0] - 1):
            cells_total_list[0] -= offset

    # y-direction
    if ndim >= 2:
        for j in range(ndom_list[1]):
            cells_total_list[1] += data_list[0][j][0].shape[1]
            if j != 0:
                cells_total_list[1] -= offset
            if j != (ndom_list[1] - 1):
                cells_total_list[1] -= offset

    # z-direction
    if ndim == 3:
        for k in range(ndom_list[2]):
            cells_total_list[2] += data_list[0][0][k].shape[2]
            if k != 0:
                cells_total_list[2] -= offset
            if k != (ndom_list[2] - 1):
                cells_total_list[2] -= offset

    return cells_list, cells_total_list


def merge_domain_data(data_list, overlap, cells_list=None, cells_total_list=None, ndom_list=None, is_ts=True):

    offset = round(overlap / 2)

    if ndom_list is None:
        ndom_list = get_nested_decomp_dims(data_list)
    if (cells_list is None) or (cells_total_list is None):
        cells_list, cells_total_list = get_full_dims_from_decomp(data_list, overlap, is_ts=is_ts)


    ndim = len(data_list[0][0][0].shape) - 1
    if is_ts:
        ndim -= 1
    varshape = data_list[0][0][0].shape[ndim:]

    data_merged = np.zeros(tuple(cells_total_list[:ndim]) + varshape, dtype=np.float64)

    ndomains = np.prod(ndom_list)
    for dom_idx in range(ndomains):

        i = dom_idx % ndom_list[0]
        j = int(dom_idx / ndom_list[0])
        k = int(dom_idx / (ndom_list[0] * ndom_list[1]))

        # data indices
        # x-direction
        if i == 0:
            start_xidx = 0
        else:
            start_xidx = int(np.sum(cells_list[0][:i, 0, 0])) - offset * (2 * i - 1)
        if i == (ndom_list[0] - 1):
            end_xidx = cells_total_list[0]
        else:
            end_xidx = int(np.sum(cells_list[0][:i+1, 0, 0])) - offset * (2 * i + 1)

        if i == 0:
            start_xidx_sub = 0
        else:
            start_xidx_sub = offset
        if i == (ndom_list[0] - 1):
            end_xidx_sub = None
        else:
            end_xidx_sub = -offset

        # y-direction
        if ndim >= 2:
            if j == 0:
                start_yidx = 0
            else:
                start_yidx = int(np.sum(cells_list[1][0, :j, 0])) - offset * (2 * j - 1)
            if j == (ndom_list[1] - 1):
                end_yidx = cells_total_list[1]
            else:
                end_yidx = int(np.sum(cells_list[1][0, :j+1, 0])) - offset * (2 * j + 1)

            if j == 0:
                start_yidx_sub = 0
            else:
                start_yidx_sub = offset
            if j == (ndom_list[1] - 1):
                end_yidx_sub = None
            else:
                end_yidx_sub = -offset

        # z-direction
        if ndim == 3:
            if k == 0:
                start_zidx = 0
            else:
                start_zidx = int(np.sum(cells_list[2][0, 0, :k])) - offset * (2 * k - 1)
            if k == (ndom_list[2] - 1):
                end_zidx = cells_total_list[2]
            else:
                end_zidx = int(np.sum(cells_list[2][0, 0, :k+1])) - offset * (2 * k + 1)

            if k == 0:
                start_zidx_sub = 0
            else:
                start_zidx_sub = offset
            if k == (ndom_list[2] - 1):
                end_zidx_sub = None
            else:
                end_zidx_sub = -offset

        # slice data and insert appropriately
        data_sub = data_list[i][j][k]
        if ndim == 1:
            data_merged[start_xidx:end_xidx, :] = data_sub[start_xidx_sub:end_xidx_sub, :]
        elif ndim == 2:
            data_merged[start_xidx:end_xidx, start_yidx:end_yidx, :] = data_sub[
                start_xidx_sub:end_xidx_sub, start_yidx_sub:end_yidx_sub, :
            ]
        else:
            data_merged[start_xidx:end_xidx, start_yidx:end_yidx, start_zidx:end_zidx, :] = data_sub[
                start_xidx_sub:end_xidx_sub, start_yidx_sub:end_yidx_sub, start_zidx_sub:end_zidx_sub, :
            ]

    return data_merged


def load_info_domain(meshdir):

    ndim = 0
    ndom_list = [1 for _ in range(3)]
    with open(os.path.join(meshdir, "info_domain.dat"), "r") as f:
        for line in f:
            label, val = line.split()
            if label == "ndomX":
                ndom_list[0] = int(val)
                ndim += 1
            elif label == "ndomY":
                ndom_list[1] = int(val)
                ndim += 1
            elif label == "ndomZ":
                ndom_list[2] = int(val)
                ndim += 1
            elif label == "overlap":
                overlap = int(val)

    assert all([ndom >= 1 for ndom in ndom_list])
    assert overlap >= 2  # TODO: adjust when non-overlapping implemented
    assert overlap % 2 == 0  # must be even for now

    return ndom_list, overlap


def load_mesh_single(meshdir):

    # get mesh dimensionality
    ndim = 0
    ncells_list = [1 for _ in range(3)]
    with open(os.path.join(meshdir, "info.dat"), "r") as f:
        for line in f:
            label, val = line.split()
            if label == "nx":
                ncells_list[0] = int(val)
                ndim += 1
            elif label == "ny":
                ncells_list[1] = int(val)
                ndim += 1
            elif label == "nz":
                ncells_list[2] = int(val)
                ndim += 1

    assert all([ncells >= 1 for ncells in ncells_list])
    ncells_list = ncells_list[:ndim]

    # get coordinates
    coords = np.loadtxt(os.path.join(meshdir, "coordinates.dat"), dtype=np.float64)
    coords = np.reshape(coords[:, 1:], tuple(ncells_list) + (ndim,), order="F")

    return coords


def load_meshes(
    meshdir,
    merge_decomp=True,
):
    """"""

    print(f"Loading mesh from {meshdir}")

    # detect decomposed vs. monolithic
    if os.path.isfile(os.path.join(meshdir, "info_domain.dat")):
        print("Decomposed mesh detected")

        # decomposition dimension
        ndom_list, overlap = load_info_domain(meshdir)
        ndomains = np.prod(ndom_list)

        coords_sub = [[[None for _ in range(ndom_list[2])] for _ in range(ndom_list[1])] for _ in range(ndom_list[0])]
        for dom_idx in range(ndomains):

            # decomposed meshes
            meshdir_sub = os.path.join(meshdir, "domain_" + str(dom_idx))
            i = dom_idx % ndom_list[0]
            j = int(dom_idx / ndom_list[0])
            k = int(dom_idx / (ndom_list[0] * ndom_list[1]))
            coords_sub[i][j][k] = load_mesh_single(meshdir_sub)

        # combine grids, if requested
        if merge_decomp:
            coords = merge_domain_data(coords_sub, overlap, is_ts=False)
        else:
            coords = None

        return coords, coords_sub

    else:
        print("Monolithic mesh detected")
        coords = load_mesh_single(meshdir)
        coords_sub = None

    return coords, coords_sub


def load_field_data_single(filename, coords, nvars):

    ndim = coords.shape[-1]
    meshdims = coords.shape[:-1]
    dofs = np.prod(meshdims) * nvars
    sol = np.fromfile(filename)
    nt = round(np.size(sol) / dofs)
    sol = np.reshape(sol, (nvars,) + meshdims + (nt,), order="F")
    sol = np.transpose(sol, tuple(np.arange(1,ndim+1)) + (ndim+1, 0, ))

    return sol


def load_field_data(
    datadir,
    fileroot,
    nvars,
    coords=None,
    meshdir=None,
    merge_decomp=True,
):
    """Loading field data from PDA binaries

    datadir: directory where binaries are stored
    fileroot: string preceding ".bin" for monolithic files, or "*_n.bin" for decomposed files
    nvars: number of state variables expected in system (e.g., 4 for 2D Euler)
    coords:
        For monolithic: single-domain coords returned from load_meshes or load_mesh_single
        For decomposed: coords_sub L-of-L-of-L returned by load_meshes, with meshes for each subdomain
    meshdir: root mesh directory
        For monolithic: required if coords not provided
        For decomposed: always required, to get decomposition info
    merge_decomp: whether to merge solution for decomposed solution
    """


    print(f"Loading data from {datadir}")

    # detect monolithic vs. decomposed
    if os.path.isfile(os.path.join(datadir, fileroot + ".bin")):

        print("Monolithic solution detected")
        if coords is None:
            assert meshdir is not None
            coords = load_mesh_single(meshdir)

        filename = os.path.join(datadir, fileroot + ".bin")
        sol = load_field_data_single(filename, coords, nvars)
        sol_sub = None

    else:

        print("Decomposed solution detected")

        assert meshdir is not None
        ndom_list, overlap = load_info_domain(meshdir)

        if coords is None:
            coords, _ = load_meshes(meshdir, merge_decomp=False)
        ndomains = np.prod(ndom_list)

        sol_sub = [[[None for _ in range(ndom_list[2])] for _ in range(ndom_list[1])] for _ in range(ndom_list[0])]
        for dom_idx in range(ndomains):

            # decomposed solutions
            filename = os.path.join(datadir, f"{fileroot}_{dom_idx}.bin")
            i = dom_idx % ndom_list[0]
            j = int(dom_idx / ndom_list[0])
            k = int(dom_idx / (ndom_list[0] * ndom_list[1]))
            sol_sub[i][j][k] = load_field_data_single(filename, coords[i][j][k], nvars)

        # combine solution, if requested
        if merge_decomp:
            sol = merge_domain_data(sol_sub, overlap, is_ts=True)
        else:
            sol = None

    return sol, sol_sub