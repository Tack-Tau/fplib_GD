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
            del_fp = np.zeros(3)
            for jat in range(len(rxyz)):
                # amp, n_sphere, rxyz_sphere, rcov_sphere = \
                # fplib_GD.get_sphere(ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, iat)
                # print ("amp", amp)
                # print ("n_sphere", n_sphere)
                # print ("rxyz_sphere", rxyz_sphere)
                # print ("rcov_sphere", rcov_sphere)
                print ("iat = ", iat)
                print ("jat = ", jat)
                D_n_i = x*iat
                D_n_j = x*jat
                fp_iat = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, iat)
                fp_jat = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, jat)
                D_fp_mat_iat = \
         fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, D_n_i, iat)
                D_fp_mat_jat = \
         fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, D_n_j, jat)
                diff_fp = fp_iat-fp_jat
                diff_D_fp = D_fp_mat_iat[:, D_n_i] - D_fp_mat_jat[:, D_n_j]
                del_fp[x] = del_fp[x] + np.dot( diff_fp,  diff_D_fp )
                if np.dot( diff_fp,  diff_D_fp ) > atol:
                    print("rxyz_final = ", rxyz)
                    sys.exit("Reached user setting tolerance, program ended")
            print ("1 rxyz", rxyz)
        print ("2 rxyz", rxyz)
        rxyz[iat][x] = rxyz[iat][x] - step_size*del_fp[x]
        print ("3 rxyz", rxyz)
        
    # print ("n_sphere", n_sphere)
    # print ("length of amp", len(amp))
    # print ("length of rcov_sphere", len(rcov_sphere))
    # print ("size of rxyz_sphere", rxyz_sphere.shape)
    print ("final rxyz", rxyz)
    return rxyz
    
        
    

if __name__ == "__main__":
    args = sys.argv
    v1 = args[1]
    test_CG(v1)
