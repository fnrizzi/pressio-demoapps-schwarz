import struct
import numpy as np

gamma = (5. + 2.) / 5.

def computePressure(rho, u, v, E):
    vel = u**2 + v**2
    return (gamma - 1.) * (E - rho * vel * 0.5)

if __name__== "__main__":
    nx = 20
    ny = 20
    fomTotDofs = nx * ny * 4

    D = np.fromfile("riemann2d_solution.bin")
    nt = int(np.size(D) / fomTotDofs)
    D = np.reshape(D, (nt, fomTotDofs))
    D = D[-1, :]
    D = np.reshape(D, (nx * ny, 4))
    rho = D[:,0]
    u   = D[:,1] / rho
    v   = D[:,2] / rho
    p   = computePressure(rho, u, v, D[:,3])
    np.savetxt("rho.txt", rho)
    np.savetxt("p.txt", p)

    goldR = np.loadtxt("rho_gold.txt")
    assert rho.shape == goldR.shape
    assert np.isnan(rho).all() == False
    assert np.isnan(goldR).all() == False
    assert np.allclose(rho, goldR, rtol=1e-10, atol=1e-12)

    goldP = np.loadtxt("p_gold.txt")
    assert p.shape == goldP.shape
    assert np.isnan(p).all() == False
    assert np.isnan(goldP).all() == False
    assert np.allclose(p, goldP, rtol=1e-10, atol=1e-12)

    # check runtime file
    f = open('runtime.bin', 'rb')
    contents = f.read()
    assert len(contents) == 16
    niters = struct.unpack('Q', contents[:8])[0]
    assert niters == 1
    timeval = struct.unpack('d', contents[8:16])[0]
    assert timeval > 0.0
