
import numpy as np
import sys, os

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
