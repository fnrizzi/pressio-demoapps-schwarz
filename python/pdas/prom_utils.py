import os
import struct

import numpy as np
from scipy.linalg import svd

from pdas.data_utils import load_meshes, load_info_domain, decompose_domain_data
from pdas.data_utils import load_unified_helper, write_to_binary, read_from_binary


def center(data_in, centervec=None, method=None):
    # data assumed to have shape (spatial, time, var)

    if centervec is None:
        assert method is not None
        if method == "zero":
            centervec = np.zeros((data_in.shape[0], 1, data_in.shape[-1]), dtype=np.float64)
        elif method == "init_cond":
            centervec = data_in[:, [0], :]
        elif (method == "mean"):
            centervec = np.mean(data_in, axis=1, keepdims=True)
        else:
            raise ValueError(f"Invalid centering method: {method}")
    else:
        # just get in shape to broadcast
        assert centervec.ndim == 2
        centervec = centervec[:, None, :]

    data_out = data_in - centervec

    return data_out, np.squeeze(centervec)


def normalize(data_in, normvec=None, method=None):
    # data assumed to have shape (spatial, time, var)

    if normvec is None:
        if method == "one":
            normvec = np.ones((data_in.shape[0], 1, data_in.shape[-1]), dtype=np.float64)
        elif method == "l2":
            normvec = np.mean(
                np.square(np.linalg.norm(data_in, axis=0, ord=2, keepdims=True)),
                axis=1,
                keepdims=True,
            ) / data_in.shape[0]
            normvec = np.repeat(normvec, data_in.shape[0], axis=0)
        else:
            raise ValueError(f"Invalid normalization method: {method}")
    else:
        # just get in shape to broadcast
        assert normvec.ndim == 2
        normvec = normvec[:, None, :]

    data_out = data_in / normvec

    return data_out, np.squeeze(normvec)


def calc_pod_single(
    data_in,
    centervec=None,
    center_method=None,
    normvec=None,
    norm_method=None,
    nmodes=None,
):
    assert (centervec is not None) or (center_method is not None)
    assert (normvec is not None) or (norm_method is not None)

    # flatten spatial dimension (I/O is column-major)
    dim = data_in.ndim - 2
    if dim == 2:
        data = np.reshape(data_in, (-1,) + data_in.shape[-2:], order="F")
    else:
        raise ValueError(f"Unsupported dimension: {dim}")

    # center, normalize data
    data_proc, centervec = center(data, centervec=centervec, method=center_method)
    data_proc, normvec = normalize(data_proc, normvec=normvec, method=norm_method)

    # TODO: could do scalar POD here
    # flatten variable dimensions
    nsamps = data_proc.shape[1]
    data_proc = np.transpose(data_proc, (0, 2, 1))
    data_proc = np.reshape(data_proc, (-1, nsamps), order="C")

    # compute POD basis, truncate
    U, _, _ = svd(data_proc, full_matrices=False)
    U = U[:, :nmodes]

    # bake normalization into basis
    U = normvec.flatten(order="F")[:, None] * U

    # return centering/normalization vectors and basis
    return U, centervec, normvec


def gen_pod_bases(
    outdir,
    meshlist=None,
    datalist=None,
    meshdir=None,
    datadir=None,
    nvars=None,
    dataroot=None,
    pod_decomp=False,
    meshdir_decomp=None,
    idx_start=0,
    idx_end=None,
    idx_skip=1,
    centervec=None,
    center_method=None,
    normvec=None,
    norm_method=None,
    nmodes=None,
):

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    meshlist, datalist = load_unified_helper(
        meshlist,
        datalist,
        meshdir,
        datadir,
        nvars,
        dataroot,
        merge_decomp=False,
    )

    # if doing decomposed POD, either must be mono solution or share mesh
    ndata_in = len(datalist)
    assert ndata_in >= 1
    if pod_decomp:

        if ndata_in == 1:
            assert meshdir_decomp is not None
            _, overlap = load_info_domain(meshdir_decomp)
            _, meshlist_decomp = load_meshes(meshdir_decomp, merge_decomp=False)

            datalist_decomp = decompose_domain_data(datalist[0], meshlist_decomp, overlap, is_ts=True, is_ts_decomp=False)

            # unroll from 3D list
            datalist = []
            for k in range(len(datalist_decomp[0][0])):
                for j in range(len(datalist_decomp[0])):
                    for i in range(len(datalist_decomp)):
                        datalist.append(datalist_decomp[i][j][k])

        else:
            # don't need do do anything, already have decomposed solution
            assert (meshlist_decomp is None) or (meshdir_decomp is None)

    ndim = datalist[0].ndim - 2

    # downsample in time
    for idx, _ in enumerate(datalist):
        if ndim == 2:
            datalist[idx] = datalist[idx][:, :, idx_start:idx_end:idx_skip, :]
        else:
            raise ValueError(f"Unsupported ndim = {ndim}")

    ndata_out = len(datalist)
    for data_idx, data in enumerate(datalist):

        # compute basis and feature scaling vectors
        basis, centervec, normvec = calc_pod_single(
            data,
            centervec=centervec,
            center_method=center_method,
            normvec=normvec,
            norm_method=norm_method,
            nmodes=nmodes,
        )

        # write to disk
        if (ndata_out == 1) and (not pod_decomp):
            numstr = ""
        else:
            numstr = f"_{data_idx}"
        basis_file = os.path.join(outdir, f"basis{numstr}.bin")
        # FIXME: this transpose and "reverse" is bad practice
        write_to_binary(basis.T, basis_file, reverse=True)
        center_file = os.path.join(outdir, f"center{numstr}.bin")
        write_to_binary(centervec.flatten(order="C"), center_file)
        norm_file = os.path.join(outdir, f"norm{numstr}.bin")
        write_to_binary(normvec.flatten(order="C"), norm_file)


def load_reduced_data(
    datadir,
    fileroot,
    nvars,
    meshdir,
    trialdir,
    centerroot,
    basisroot,
    nmodes,
):

    print(f"Loading data from {datadir}")

    # detect monolithic vs decomposed
    if os.path.isfile(os.path.join(datadir, fileroot + ".bin")):

        print("Monolithic solution detected")
        coords, coords_sub = load_meshes(meshdir)
        assert coords_sub is None
        ndim = coords.shape[-1]
        meshdims = coords.shape[:-1]

        center_file = os.path.join(trialdir, centerroot + ".bin")
        center = read_from_binary(center_file)
        basis_file = os.path.join(trialdir, basisroot + ".bin")
        basis = read_from_binary(basis_file)[:, :nmodes]

        data_red = np.fromfile(os.path.join(datadir, fileroot + ".bin"))
        data_red = np.reshape(data_red, (nmodes, -1), order="F")

        data_full = center + basis @ data_red
        nsnaps = data_full.shape[-1]
        data_full = np.reshape(data_full, ((nvars,) + meshdims + (nsnaps,)), order="F")
        data_full = np.transpose(data_full, tuple(np.arange(1,ndim+1)) + (ndim+1, 0,))

    else:

        raise ValueError

    return data_full
