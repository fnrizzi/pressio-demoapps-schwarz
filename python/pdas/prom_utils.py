import os
import struct

import numpy as np
from scipy.linalg import svd

from pdas.data_utils import load_meshes, load_info_domain, decompose_domain_data
from pdas.data_utils import load_unified_helper, write_to_binary


def center(data_in, centervec=None, method=None):
    # data assumed to have shape (spatial, time, var)

    if centervec is None:
        assert method is not None
        if method == "init_cond":
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
    data_proc = np.transpose(data_proc, (2, 0, 1))
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
    ndata = len(datalist)
    assert ndata >= 1
    if pod_decomp:

        if ndata == 1:
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
        basis_file = os.path.join(outdir, f"basis_{data_idx}.bin")
        write_to_binary(basis, basis_file)
        center_file = os.path.join(outdir, f"center_{data_idx}.bin")
        write_to_binary(centervec.flatten(order="F"), center_file)
        norm_file = os.path.join(outdir, f"norm_{data_idx}.bin")
        write_to_binary(normvec.flatten(order="F"), norm_file)

