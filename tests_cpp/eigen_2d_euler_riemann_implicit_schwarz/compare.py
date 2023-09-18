import struct
import numpy as np

gamma = (5.+2.)/5.

def computePressure(rho, u, v, E):
  vel = u**2 + v**2
  return (gamma - 1.) * (E - rho*vel*0.5)

if __name__== "__main__":

    nx = 30
    ny = 30
    fomTotDofs = nx * ny * 4

    allclose_rho = []
    allclose_p   = []
    for dom_idx in range(4):
        D = np.fromfile(f"riemann2d_solution_{dom_idx}.bin")
        nt = int(np.size(D) / fomTotDofs)
        D = np.reshape(D, (nt, fomTotDofs))
        D = D[-1, :]
        D = np.reshape(D, (nx * ny, 4))
        rho = D[:, 0]
        u   = D[:, 1] / rho
        v   = D[:, 2] / rho
        p   = computePressure(rho, u, v, D[:,3])
        np.savetxt(f"rho_{dom_idx}.txt", rho)
        np.savetxt(f"p_{dom_idx}.txt", p)

        goldR = np.loadtxt(f"rho_gold_{dom_idx}.txt")
        assert rho.shape == goldR.shape
        assert np.isnan(rho).all() == False
        assert np.isnan(goldR).all() == False
        allclose_rho.append(np.allclose(rho, goldR, rtol=1e-10, atol=1e-12))

        goldP = np.loadtxt(f"p_gold_{dom_idx}.txt")
        assert p.shape == goldP.shape
        assert np.isnan(p).all() == False
        assert np.isnan(goldP).all() == False
        allclose_p.append(np.allclose(p, goldP, rtol=1e-10, atol=1e-12))

    assert all(allclose_rho)
    assert all(allclose_p)

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

    assert niters == 200
