import numpy as np
import fplib_GD
import sys

# Move function `readvasp(vp)` from test set to `fplib_FD.py`

def test4(v1, v2):
    ntyp = 1
    nx = 300
    lmax = 0
    cutoff = 6.5
    znucl = np.array([3], int)
    lat1, rxyz1, types = fplib_GD.readvasp(v1)
    lat2, rxyz2, types = fplib_GD.readvasp(v2)
    contract = False
    iter_max = 100
    atol = 1e-3
    res_CG1 = fplib_GD.get_fpCG(contract, ntyp, nx, lmax, lat1, rxyz1, types, znucl, cutoff)
    res_CG2 = fplib_GD.get_fpCG(contract, ntyp, nx, lmax, lat2, rxyz2, types, znucl, cutoff)
    print ('First fingerprint gradient descent: ', res_CG1)
    print ('Second fingerprint gradient descent: ', res_CG2)


if __name__ == "__main__":
    args = sys.argv
    v1 = args[1]
    v2 = args[2]
    test4(v1, v2)
