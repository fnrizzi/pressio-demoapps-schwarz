import os
import struct
from math import floor

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

    if ndom_list is None:
        ndom_list = get_nested_decomp_dims(data_list)

    # coordinate data should have ndim + 1 (spatial dim) dimensions
    # state data should have ndim + 2 (time, variable) dimensions
    ndim = len(data_list[0][0][0].shape) - 1
    if is_ts:
        ndim -= 1

    # combining grid dimensions, if requested
    cells_list = [np.zeros(tuple([ndom for ndom in ndom_list]), dtype=np.int32) for _ in range(3)]
    cells_total_list = [0 for _ in range(3)]

    # count number of cells that were added to each side of a subdomain
    added_bound = [[[0, 0] for _ in range(ndom_list[dim])] for dim in range(3)]
    for dim in range(ndim):
        ndom = ndom_list[dim]
        tot_overlap = overlap * (ndom - 1)
        added = floor(tot_overlap / ndom)
        extra = tot_overlap % ndom
        for dom_idx in range(ndom):

            to_add = added
            if dom_idx >= (ndom - extra):
                to_add += 1

            if dom_idx == 0:
                added_bound[dim][dom_idx][0] = 0
                added_bound[dim][dom_idx][1] = added
            else:
                added_bound[dim][dom_idx][0] = overlap - added_bound[dim][dom_idx-1][1]
                added_bound[dim][dom_idx][1] = to_add - added_bound[dim][dom_idx][0]

    # individual domain shapes
    for i in range(ndom_list[0]):
        for j in range(ndom_list[1]):
            for k in range(ndom_list[2]):
                cells = list(data_list[i][j][k].shape)
                for dim in range(ndim):
                    cells_list[dim][i, j, k] = cells[dim] - sum(added_bound[dim][[i, j, k][dim]])

    # x-direction
    for i in range(ndom_list[0]):
        cells_total_list[0] += data_list[i][0][0].shape[0] - sum(added_bound[0][i])

    # y-direction
    if ndim >= 2:
        for j in range(ndom_list[1]):
            cells_total_list[1] += data_list[0][j][0].shape[1] - sum(added_bound[1][j])

    # z-direction
    if ndim == 3:
        for k in range(ndom_list[2]):
            cells_total_list[2] += data_list[0][0][k].shape[2] - sum(added_bound[2][k])

    return cells_list, cells_total_list, added_bound


def merge_domain_data(data_list, overlap, is_ts=True):

    ndom_list = get_nested_decomp_dims(data_list)
    cells_list, cells_total_list, added_bound = get_full_dims_from_decomp(data_list, overlap, is_ts=is_ts)

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
        start_xidx = np.sum(cells_list[0][:i, 0, 0])
        end_xidx = np.sum(cells_list[0][:i+1, 0, 0])
        start_xidx_sub = added_bound[0][i][0]
        end_xidx_sub = start_xidx_sub + cells_list[0][i, 0, 0]

        # y-direction
        if ndim >= 2:

            start_yidx = np.sum(cells_list[1][0, :j, 0])
            end_yidx = np.sum(cells_list[1][0, :j+1, 0])
            start_yidx_sub = added_bound[1][j][0]
            end_yidx_sub = start_yidx_sub + cells_list[1][0, j, 0]

        # z-direction
        if ndim == 3:

            start_zidx = np.sum(cells_list[2][0, 0, :k])
            end_zidx = np.sum(cells_list[2][0, 0, :k+1])
            start_zidx_sub = added_bound[2][k][0]
            end_zidx_sub = start_zidx_sub + cells_list[2][0, 0, k]

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


def decompose_domain_data(
    data_single,
    decomp_list,
    overlap,
    is_ts=True,
    is_ts_decomp=True,
):

    ndom_list = get_nested_decomp_dims(decomp_list)
    cells_list, cells_total_list, added_bound = get_full_dims_from_decomp(decomp_list, overlap, is_ts=is_ts_decomp)

    ndim = len(data_single.shape) - 1
    if is_ts:
        ndim -= 1

    # check that total dimensions line up
    assert all([data_single.shape[dim] == cells_total_list[dim] for dim in range(ndim)])

    data_decomp = [[[None for _ in range(ndom_list[2])] for _ in range(ndom_list[1])] for _ in range(ndom_list[0])]

    ndomains = np.prod(ndom_list)
    for dom_idx in range(ndomains):

        i = dom_idx % ndom_list[0]
        j = int(dom_idx / ndom_list[0])
        k = int(dom_idx / (ndom_list[0] * ndom_list[1]))

        # x-direction
        start_xidx = np.sum(cells_list[0][:i, 0, 0]) - added_bound[0][i][0]
        end_xidx = np.sum(cells_list[0][:i+1, 0, 0]) + added_bound[0][i][1]

        # y-direction
        if ndim >= 2:
            start_yidx = np.sum(cells_list[1][0, :j, 0]) - added_bound[1][j][0]
            end_yidx = np.sum(cells_list[1][0, :j+1, 0]) + added_bound[1][j][1]

        # z-direction
        if ndim == 3:
            start_zidx = np.sum(cells_list[2][0, 0, :k]) - added_bound[2][k][0]
            end_zidx = np.sum(cells_list[2][0, 0, :k+1]) + added_bound[2][k][1]

        if ndim == 1:
            data_decomp[i][j][k] = data_single[start_xidx:end_xidx, :]
        elif ndim == 2:
            data_decomp[i][j][k] = data_single[start_xidx:end_xidx, start_yidx:end_yidx, :]
        else:
            data_decomp[i][j][k] = data_single[start_xidx:end_xidx, start_yidx:end_yidx, start_zidx:end_zidx, :]

    return data_decomp

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
    assert overlap > 1  # TODO: adjust when non-overlapping implemented

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


def calc_mesh_bounds(meshdirs):

    if isinstance(meshdirs, str):
        meshdirs = [meshdirs]
    assert isinstance(meshdirs, list)
    nmeshes = len(meshdirs)

    bounds = [None for _ in range(nmeshes)]
    for mesh_idx, meshdir in enumerate(meshdirs):
        # check if it's a decomposed mesh
        if os.path.isfile(os.path.join(meshdir, "info_domain.dat")):
            is_decomp = True
        else:
            is_decomp = False

        # load mesh
        coords, coords_sub = load_meshes(meshdir, merge_decomp=False)
        if is_decomp:
            bounds[mesh_idx] = []
            for k in range(len(coords_sub[0][0])):
                for j in range(len(coords_sub[0])):
                    for i in range(len(coords_sub)):
                        ndim = coords_sub[i][j][k].shape[-1]
                        if ndim == 2:
                            x = coords_sub[i][j][k][:, :, 0]
                            y = coords_sub[i][j][k][:, :, 1]
                            bounds[mesh_idx].append([
                                [np.amin(x), np.amax(x)],
                                [np.amin(y), np.amax(y)],
                            ])
                        else:
                            raise ValueError(f"Invalid dimension: {ndim}")

    return bounds


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
        coords = load_mesh_single(meshdir)
        coords_sub = None
        print("Monolithic mesh detected")

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

        assert meshdir is not None
        ndom_list, overlap = load_info_domain(meshdir)

        print("Decomposed solution detected")

        if coords is None:
            _, coords = load_meshes(meshdir, merge_decomp=False)
            assert coords is not None
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


def load_unified_helper(
    meshlist=None,
    datalist=None,
    meshdirs=None,
    datadirs=None,
    nvars=None,
    dataroot=None,
    merge_decomp=True,
):
    # helper function for other functions that need to load mesh and data
    # checks inputs, returns list of data/meshes


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

    return meshlist, datalist


def euler_calc_pressure(
    gamma,
    meshlist=None,
    datalist=None,
    meshdirs=None,
    datadirs=None,
    nvars=None,
    dataroot=None,
    merge_decomp=True,
):

    _, datalist = load_unified_helper(
        meshlist,
        datalist,
        meshdirs,
        datadirs,
        nvars,
        dataroot,
        merge_decomp=merge_decomp,
    )

    pressurelist = [None for _ in range(len(datalist))]
    for data_idx, data in enumerate(datalist):

        ndim = data.ndim - 2
        if ndim == 2:
            mom_mag = np.sum(np.square(data[:, :, :, 1:3]), axis=3)
            pressurelist[data_idx] = (gamma - 1.0) * (data[:, :, :, 3] - 0.5 * mom_mag / data[:, :, :, 0])[:, :, :, None]
        else:
            raise ValueError(f"Invalid dimension: {ndim}")

    return pressurelist


# FIXME: This use of "reverse" is bad, need to figure out proper write order
def write_to_binary(data, outfile, reverse=False):

    nrows = data.shape[0]
    if data.ndim == 1:
        ncols = 1
    elif data.ndim == 2:
        ncols = data.shape[1]
    else:
        raise ValueError(f"Unexpected array ndim: {data.ndim}")

    if reverse:
        temp = nrows
        nrows = ncols
        ncols = temp

    # NOTE: demoapps reads header as size_t, which is 8-byte
    with open(outfile, "wb") as f:
        np.array([nrows], dtype=np.int64).tofile(f)
        np.array([ncols], dtype=np.int64).tofile(f)
        data.astype(np.float64).tofile(f)


def read_from_binary(infile):

    with open(infile, "rb") as f:
        contents = f.read()

    m, n = struct.unpack("QQ", contents[:16])
    data = struct.unpack("d"*m*n, contents[16:])
    data = np.reshape(np.array(data), (m, n), "F")

    return data


def read_runtimes(
    datadirs,
    dataroot,
    methodlist,
):

    if isinstance(datadirs, str):
        datadirs = [datadirs]
    ndata = len(datadirs)

    assert len(methodlist) == ndata
    assert all([method in ["mono", "mult", "add"] for method in methodlist])

    runtimelist = [None for _ in range(ndata)]
    niterslist = [None for _ in range(ndata)]
    for data_idx, datadir in enumerate(datadirs):

        datafile = os.path.join(datadir, f"{dataroot}.bin")
        with open(datafile, 'rb') as f:
            contents = f.read()

        ndomains = struct.unpack('Q', contents[:8])[0]
        nbytes_file = len(contents)
        nbytes_read = 8

        timelist = []
        niters_tot = 0
        while nbytes_read < nbytes_file:

            niters = struct.unpack('Q', contents[nbytes_read:nbytes_read+8])[0]
            nbytes_read += 8
            niters_tot += niters
            runtime_arr = np.zeros((ndomains, niters), dtype=np.float64)

            for iter_idx in range(niters):
                runtime_vals = struct.unpack('d'*ndomains, contents[nbytes_read:nbytes_read+8*ndomains])
                runtime_arr[:, iter_idx] = np.array(runtime_vals, dtype=np.float64)
                nbytes_read += 8*ndomains

            timelist.append(runtime_arr.copy())

        niterslist[data_idx] = niters_tot

        # single-domain
        if (len(timelist) == 1) and (timelist[0].shape == (1, 1)):
            assert methodlist[data_idx] == "mono"
            runtimelist[data_idx] = timelist[0][0, 0]

        else:
            assert methodlist[data_idx] in ["mult", "add"]
            runtime_est = 0.0
            for runtime_arr in timelist:
                if methodlist[data_idx] == "mult":
                    # multiplicative
                    runtime_est += np.sum(runtime_arr)
                else:
                    # additive
                    runtime_est += np.sum(np.amax(runtime_arr, axis=0))
            runtimelist[data_idx] = runtime_est

    return runtimelist, niterslist
