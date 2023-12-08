import struct
import numpy as np

if __name__== "__main__":
    nx = 30
    ny = 30
    fomTotDofs = nx * ny * 3

    allclose = []
    for dom_idx in range(4):
        D = np.fromfile(f"swe_slipWall2d_solution_{dom_idx}.bin")
        nt = int(np.size(D) / fomTotDofs)
        D = np.reshape(D, (nt, fomTotDofs))
        D = D[-1, :]
        D = np.reshape(D, (nx * ny, 3))
        h = D[:, 0]
        np.savetxt(f"h_{dom_idx}.txt", h)

        goldD = np.loadtxt(f"h_gold_{dom_idx}.txt")
        assert h.shape == goldD.shape
        assert np.isnan(h).all() == False
        assert np.isnan(goldD).all() == False
        allclose.append(np.allclose(h, goldD, rtol=1e-10, atol=1e-12))

    assert all(allclose)

    # check runtime file
    f = open('runtime.bin', 'rb')
    contents = f.read()
    nbytes_file = len(contents)
    assert nbytes_file > 0
    ndomains = struct.unpack('Q', contents[:8])[0]
    assert ndomains == 4

    nbytes_read = 8
    niters = 0
    while nbytes_read < nbytes_file:

        nsubiters = struct.unpack('Q', contents[nbytes_read:nbytes_read+8])[0]
        nbytes_read += 8
        runtime_arr = np.zeros((ndomains, nsubiters), dtype=np.float64)

        for iter_idx in range(nsubiters):
            runtime_vals = struct.unpack('d'*ndomains, contents[nbytes_read:nbytes_read+8*ndomains])
            runtime_arr[:, iter_idx] = np.array(runtime_vals, dtype=np.float64)
            nbytes_read += 8*ndomains

        assert np.amin(runtime_arr) > 0.0

        niters += 1

    assert niters == 500

