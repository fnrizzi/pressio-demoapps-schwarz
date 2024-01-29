
import struct
import numpy as np


if __name__== "__main__":

    nmodes = 25

    D = np.fromfile("swe_slipWall2d_solution.bin")
    nt = int(np.size(D)/nmodes)
    D = np.reshape(D, (nt, nmodes))
    D = D[-1, :]
    np.savetxt("reducedState.txt", D)

    goldD = np.loadtxt("reducedState_gold.txt")
    assert(np.allclose(D.shape, goldD.shape))
    assert(np.isnan(D).all() == False)
    assert(np.isnan(goldD).all() == False)
    assert(np.allclose(D, goldD,rtol=1e-10, atol=1e-12))

    # check runtime file
    f = open('runtime.bin', 'rb')
    contents = f.read()
    assert len(contents) == 16
    niters = struct.unpack('Q', contents[:8])[0]
    assert niters == 1
    timeval = struct.unpack('d', contents[8:16])[0]
    assert timeval > 0.0