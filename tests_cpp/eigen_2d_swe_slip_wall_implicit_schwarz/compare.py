
import numpy as np

if __name__== "__main__":
  nx = 30
  ny = 30
  fomTotDofs = nx*ny*3

  for dom_idx in range(4):
    D = np.fromfile(f"swe_slipWall2d_solution_{dom_idx}.bin")
    nt = int(np.size(D)/fomTotDofs)
    D = np.reshape(D, (nt, fomTotDofs))
    D = D[-1, :]
    D = np.reshape(D, (nx*ny, 3))
    h = D[:,0]
    np.savetxt(f"h_{dom_idx}.txt", h)

    goldD = np.loadtxt(f"h_gold_{dom_idx}.txt")
    assert(h.shape == goldD.shape)
    assert(np.isnan(h).all() == False)
    assert(np.isnan(goldD).all() == False)
    assert(np.allclose(h, goldD, rtol=1e-10, atol=1e-12))

