import struct
import numpy as np

if __name__== "__main__":
    nx = 90
    ny = 100
    fomTotDofs = nx * ny * 3

    D = np.fromfile("swe_slipWall2d_solution.bin")
    nt = int(np.size(D) / fomTotDofs)
    D = np.reshape(D, (nt, fomTotDofs))
    np.savetxt("solution_full.txt", D.T)
    D = D[-1, :]
    D = np.reshape(D, (nx*ny, 3))
    h = D[:, 0]
    np.savetxt("h.txt", h)

    goldD = np.loadtxt("h_gold.txt")
    assert np.allclose(h.shape, goldD.shape)
    assert np.isnan(h).all() == False
    assert np.isnan(goldD).all() == False
    assert np.allclose(h, goldD, rtol=1e-10, atol=1e-12)

    # check runtime file
    f = open('runtime.bin', 'rb')
    contents = f.read()
    assert len(contents) == 16
    niters = struct.unpack('Q', contents[:8])[0]
    assert niters == 1
    timeval = struct.unpack('d', contents[8:16])[0]
    assert timeval > 0.0
