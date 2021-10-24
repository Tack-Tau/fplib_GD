import numpy as np
import fplib_GD
import sys

# Move function `readvasp(vp)` from test set to `fplib_FD.py`

def test_CG(v1):
    ntyp = 1
    nx = 300
    lmax = 0
    cutoff = 6.5
    znucl = np.array([3], int)
    lat, rxyz, types = fplib_GD.readvasp(v1)
    contract = False
    iter_max = 100
    atol = 1e-6
    step_size = 1e-4
    for x in range(3):
        for iat in range(len(rxyz)):
        del_fp = 0.0
            for jat in range(len(rxyz)):
                fp_iat = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, iat)
                fp_jat = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, jat)
                D_fp_mat_iat = \
           fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, D_n, iat)
                D_fp_mat_jat = \
           fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, D_n, jat)
                diff_fp = fp_iat-fp_jat
                diff_D_fp = D_fp_mat_iat[:, iat-1] - D_fp_mat_jat[:, jat-1]
                del_fp = del_fp + np.dot( diff_fp,  diff_D_fp )
        rxyz[iat] = rxyz[iat] - step_size*del_fp
    

if __name__ == "__main__":
    args = sys.argv
    v1 = args[1]
    test_CG(v1)
