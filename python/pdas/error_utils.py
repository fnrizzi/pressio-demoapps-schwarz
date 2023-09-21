
import numpy as np

from pdas.data_utils import load_unified_helper


def calc_shared_samp_interval(
    dtlist=None,
    samplist=None,
    ndata=1,
):
    if (dtlist is not None) and (samplist is not None):
        if isinstance(dtlist, list):
            assert len(dtlist) == ndata
            dtlist = np.array(dtlist, dtype=np.float64)
        else:
            dtlist = dtlist * np.ones(ndata, dtype=np.float64)
        if isinstance(samplist, list):
            assert len(samplist) == ndata
            samplist = np.array(samplist, dtype=np.float64)
        else:
            samplist = samplist * np.ones(ndata, dtype=np.float64)

        # make sure there's a universally shared interval
        samplengths = dtlist * samplist
        sampintervals = np.amax(samplengths) / samplengths
        assert all([samp.is_integer() for samp in sampintervals])
        sampintervals = sampintervals.astype(np.int32)

    else:
        sampintervals = np.ones(ndata, dtype=np.int32)

    return samplengths, sampintervals

def calc_error_fields(
    meshlist=None,
    datalist=None,
    meshdirs=None,
    datadirs=None,
    nvars=None,
    dataroot=None,
    dtlist=None,
    samplist=None,
):
    # NOTE: the FIRST data datalist/datadirs element is treated as "truth"

    # always merge decompositions for ease
    _, datalist = load_unified_helper(
        meshlist=meshlist,
        datalist=datalist,
        meshdirs=meshdirs,
        datadirs=datadirs,
        nvars=nvars,
        dataroot=dataroot,
        merge_decomp=True,
    )
    ndata = len(datalist)
    assert ndata > 1
    ndim = datalist[0].ndim - 2

    # If samplist and dtlist provided, comparison interval is explicit
    # Otherwise, same dt and sampling interval assumed
    samplengths, sampintervals = calc_shared_samp_interval(
        dtlist=dtlist,
        samplist=samplist,
        ndata=ndata,
    )

    # compute SIGNED errors (comparison - truth, not absolute)
    errorlist = []
    for data_idx in range(1, ndata):

        if ndim == 2:
            errorlist.append(
                datalist[data_idx][:, :, ::sampintervals[data_idx], :] - \
                datalist[0][:, :, ::sampintervals[0], :]
            )
        else:
            raise ValueError(f"Invalid ndim: {ndim}")

        # double check that everything matches up
        if data_idx == 1:
            nsamps = errorlist[-1].shape[-2]
        else:
            assert errorlist[-1].shape[-2] == nsamps

    # get sample times (ignoring any possible offset), for later plotting
    if (dtlist is not None) and (samplist is not None):
        samptimes = np.arange(nsamps) * np.amax(samplengths)
    else:
        samptimes = np.arange(nsamps)

    return errorlist, samptimes

def calc_error_norms(
    errorlist=None,
    samptimes=None,
    meshlist=None,
    datalist=None,
    meshdirs=None,
    datadirs=None,
    nvars=None,
    dataroot=None,
    dtlist=None,
    samplist=None,
    timenorm=False,
    spacenorm=False,
    relative=False,
):
    assert timenorm or spacenorm

    # if computing relative norm, need data for denominator
    if relative:
        # this assert is a little weird, but don't want to potentially mix up error fields from different sources
        # can see an instance of passing an error list that doesn't correspond to the same datalist
        assert errorlist is None

        _, datalist = load_unified_helper(
            meshlist=meshlist,
            datalist=datalist,
            meshdirs=meshdirs,
            datadirs=datadirs,
            nvars=nvars,
            dataroot=dataroot,
            merge_decomp=True,
        )
        # only need the truth value
        datatruth = datalist[0]

    if errorlist is None:
        errorlist, samptimes = calc_error_fields(
            meshlist=meshlist,
            datalist=datalist,
            meshdirs=meshdirs,
            datadirs=datadirs,
            nvars=nvars,
            dataroot=dataroot,
            dtlist=dtlist,
            samplist=samplist,
        )
        nsamps = samptimes.shape[0]
    else:
        if not isinstance(errorlist, list):
            errorlist = [errorlist]
        nsamps = errorlist[0].shape[-2]
        assert all([error.shape[-2] == nsamps for error in errorlist])
        if samptimes is None:
            samptimes = np.arange(nsamps)
        else:
            assert samptimes.shape[0] == nsamps

    ndim = errorlist[0].ndim - 2
    space_axes = tuple(range(ndim))
    time_axis = ndim

    # need same sampling rate for time norm
    if relative and timenorm:
        _, sampintervals = calc_shared_samp_interval(
            dtlist=dtlist,
            samplist=samplist,
            ndata=len(datalist),
        )

    # relative error scaling factors
    if relative:
        if timenorm:
            relfacs = np.linalg.norm(datatruth[:, :, ::sampintervals[0], :], ord=2, axis=time_axis, keepdims=True)
        else:
            relfacs = datatruth.copy()
        if spacenorm:
            relfacs = np.linalg.norm(relfacs, ord=2, axis=space_axes, keepdims=True)
    else:
        relfacs = 1.0

    # compute norms
    for error_idx, error in enumerate(errorlist):

        if timenorm:
            error = np.linalg.norm(error, ord=2, axis=time_axis, keepdims=True)
        if spacenorm:
            error = np.linalg.norm(error, ord=2, axis=space_axes, keepdims=True)
        errorlist[error_idx] = np.squeeze(error / relfacs)

    return errorlist, samptimes