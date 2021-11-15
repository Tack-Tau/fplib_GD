import numpy as np
import fplib_GD
import sys
np.random.seed(42)
# Move function `readvasp(vp)` from test set to `fplib_FD.py`


#Calculate crystal atomic finger print force and steepest descent update
def test1_CG(v1):
    ntyp = 1
    nx = 300
    lmax = 0
    cutoff = 6.5
    znucl = np.array([3], int)
    lat, rxyz, types = fplib_GD.readvasp(v1)
    contract = False
    i_iter = 0
    iter_max = 4
    atol = 1e-6
    step_size = 1e-4
    const_factor = 1.0e+38
    rxyz_new = rxyz.copy()
    
    for i_iter in range(iter_max+1):
        # del_fp = np.zeros(3)
        for i_atom in range(len(rxyz_new)):
            del_fp = np.zeros(3)
            for j_atom in range(len(rxyz_new)):
                fp_iat = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, i_atom)
                fp_jat = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, j_atom)
                D_fp_mat_iat = \
                fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, i_atom)
                D_fp_mat_jat = \
                fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, j_atom)
                diff_fp = fp_iat-fp_jat
                # common_count, i_rxyz_sphere_1, i_rxyz_sphere_2 = \
                # fplib_GD.get_common_sphere(ntyp, nx, lmax, lat, rxyz, types, \
                #                                 znucl, cutoff, i_atom, j_atom)
                iat_in_j_sphere, iat_j = fplib_GD.get_common_sphere(ntyp, \
                              nx, lmax, lat, rxyz_new, types, znucl, cutoff, i_atom, j_atom)
                if iat_in_j_sphere:
                    diff_D_fp_x = D_fp_mat_iat[0, :, i_atom] - D_fp_mat_jat[0, :, iat_j]
                    diff_D_fp_y = D_fp_mat_iat[1, :, i_atom] - D_fp_mat_jat[1, :, iat_j]
                    diff_D_fp_z = D_fp_mat_iat[2, :, i_atom] - D_fp_mat_jat[2, :, iat_j]
                else:
                    diff_D_fp_x = D_fp_mat_iat[0, :, i_atom]
                    diff_D_fp_y = D_fp_mat_iat[1, :, i_atom]
                    diff_D_fp_z = D_fp_mat_iat[2, :, i_atom]
                del_fp[0] = del_fp[0] + const_factor*np.matmul( diff_fp.T,  diff_D_fp_x )
                del_fp[1] = del_fp[1] + const_factor*np.matmul( diff_fp.T,  diff_D_fp_y )
                del_fp[2] = del_fp[2] + const_factor*np.matmul( diff_fp.T,  diff_D_fp_z )
                
                
                
                '''
                for i_common in range(common_count):
                    diff_D_fp_x = D_fp_mat_iat[0, :, i_rxyz_sphere_1[i_common]] - \
                                  D_fp_mat_jat[0, :, i_rxyz_sphere_2[i_common]]
                    diff_D_fp_y = D_fp_mat_iat[1, :, i_rxyz_sphere_1[i_common]] - \
                                  D_fp_mat_jat[1, :, i_rxyz_sphere_2[i_common]]
                    diff_D_fp_z = D_fp_mat_iat[2, :, i_rxyz_sphere_1[i_common]] - \
                                  D_fp_mat_jat[2, :, i_rxyz_sphere_2[i_common]]
                    del_fp[0] = del_fp[0] + np.matmul( diff_fp.T,  diff_D_fp_x )
                    del_fp[1] = del_fp[1] + np.matmul( diff_fp.T,  diff_D_fp_y )
                    del_fp[2] = del_fp[2] + np.matmul( diff_fp.T,  diff_D_fp_z )
                # if i_atom in i_rxyz_sphere_1 and j_atom in i_rxyz_sphere_2:
                #     diff_D_fp_x = D_fp_mat[0, :, i_atom] - D_fp_mat[0, :, j_atom]
                #     diff_D_fp_y = D_fp_mat[1, :, i_atom] - D_fp_mat[1, :, j_atom]
                #     diff_D_fp_z = D_fp_mat[2, :, i_atom] - D_fp_mat[2, :, j_atom]
                # else:
                #     diff_D_fp_x = np.zeros_like(diff_fp)
                #     diff_D_fp_y = np.zeros_like(diff_fp)
                #     diff_D_fp_z = np.zeros_like(diff_fp)
                # diff_D_fp_x = D_fp_mat[0, :, i_atom] - D_fp_mat[0, :, j_atom]
                # diff_D_fp_y = D_fp_mat[1, :, i_atom] - D_fp_mat[1, :, j_atom]
                # diff_D_fp_z = D_fp_mat[2, :, i_atom] - D_fp_mat[2, :, j_atom]
                # del_fp[0] = del_fp[0] + np.matmul( diff_fp.T,  diff_D_fp_x )
                # del_fp[1] = del_fp[1] + np.matmul( diff_fp.T,  diff_D_fp_y )
                # del_fp[2] = del_fp[2] + np.matmul( diff_fp.T,  diff_D_fp_z )
                '''
                
                
                # print("del_fp = ", del_fp)
                # rxyz[i_atom] = rxyz[i_atom] - step_size*del_fp
                '''
                if max(del_fp) < atol:
                    print ("i_iter = {0:d} \nrxyz_final = \n{1:s}".\
                          format(i_iter, np.array_str(rxyz, precision=6, suppress_small=False)))
                    return
                    # with np.printoptions(precision=3, suppress=True):
                    # sys.exit("Reached user setting tolerance, program ended")
                else:
                    print ("i_iter = {0:d} \nrxyz = \n{1:s}".\
                          format(i_iter, np.array_str(rxyz, precision=6, suppress_small=False)))
                '''
            
            rxyz_new[i_atom] = rxyz_new[i_atom] - step_size*del_fp
            print ("i_iter = {0:d} \nrxyz_final = \n{1:s}".\
                  format(i_iter, np.array_str(rxyz_new, precision=6, suppress_small=False)))
    
    
    '''
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
                # D_n_i = x*iat
                # D_n_j = x*jat
                fp_iat = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff, iat)
                fp_jat = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff, jat)
                D_fp_mat_iat = \
                fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff)
                D_fp_mat_jat = \
                fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff)
                diff_fp = fp_iat-fp_jat
                diff_D_fp = D_fp_mat_iat[:, D_n_i] - D_fp_mat_jat[:, D_n_j]
                del_fp[x] = del_fp[x] + np.dot( diff_fp,  diff_D_fp )
                print ("force in x direction", x, -del_fp[x])
                if np.dot( diff_fp,  diff_D_fp ) < atol:
                    print("rxyz_final = \n", rxyz)
                    print("Reached user setting tolerance, program ended")
            # print ("1 rxyz", rxyz)
        # print ("2 rxyz", rxyz)
        rxyz[iat][x] = rxyz[iat][x] - step_size*del_fp[x]
        # print ("3 rxyz", rxyz)
    
    # print ("n_sphere", n_sphere)
    # print ("length of amp", len(amp))
    # print ("length of rcov_sphere", len(rcov_sphere))
    # print ("size of rxyz_sphere", rxyz_sphere.shape)
    
    return rxyz
    '''

# Calculate crystal atomic finger print energy
def test2_CG(v1):
    ntyp = 1
    nx = 300
    lmax = 0
    cutoff = 6.5
    znucl = np.array([3], int)
    lat, rxyz, types = fplib_GD.readvasp(v1)
    contract = False
    fp_dist = 0.0
    for ityp in range(ntyp):
        itype = ityp + 1
        for iat in range(len(rxyz)):
            if types[iat] == itype:
                for jat in range(len(rxyz)):
                    if types[jat] == itype:
                        fp_iat = \
                        fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff, iat)
                        fp_jat = \
                        fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff, jat)
                        fp_dist = fp_dist + fplib_GD.get_fpdist(ntyp, types, fp_iat, fp_jat)

    
    print ( "Finger print energy = {0:s}".format(np.array_str(fp_dist, \
                                               precision=6, suppress_small=False)) )
    # return fp_dist



def test3_CG(v1):
    ntyp = 1
    nx = 300
    lmax = 0
    cutoff = 6.5
    znucl = np.array([3], int)
    lat, rxyz, types = fplib_GD.readvasp(v1)
    contract = False
    i_iter = 0
    iter_max = 4
    atol = 1.0e-6
    step_size = 1e-4
    const_factor = 1.0e+31
    del_fp_dist = 0.0
    rxyz_new = rxyz.copy()
    for i_iter in range(iter_max+1):
        rxyz_delta = fplib_GD.get_rxyz_delta(rxyz)
        rxyz_new = np.add(rxyz_new, rxyz_delta)
        for i_atom in range(len(rxyz)):
            del_fp = np.zeros(3)
            for j_atom in range(len(rxyz)):
                fp_iat = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, i_atom)
                fp_jat = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, j_atom)
                D_fp_mat_iat = \
                fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, i_atom)
                D_fp_mat_jat = \
                fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, j_atom)
                diff_fp = fp_iat-fp_jat
                # common_count, i_rxyz_sphere_1, i_rxyz_sphere_2 = \
                # fplib_GD.get_common_sphere(ntyp, nx, lmax, lat, rxyz, types, \
                #                                 znucl, cutoff, i_atom, j_atom)
                iat_in_j_sphere, iat_j = fplib_GD.get_common_sphere(ntyp, \
                              nx, lmax, lat, rxyz_new, types, znucl, cutoff, i_atom, j_atom)
                if iat_in_j_sphere:
                    diff_D_fp_x = D_fp_mat_iat[0, :, i_atom] - D_fp_mat_jat[0, :, iat_j]
                    diff_D_fp_y = D_fp_mat_iat[1, :, i_atom] - D_fp_mat_jat[1, :, iat_j]
                    diff_D_fp_z = D_fp_mat_iat[2, :, i_atom] - D_fp_mat_jat[2, :, iat_j]
                else:
                    diff_D_fp_x = D_fp_mat_iat[0, :, i_atom]
                    diff_D_fp_y = D_fp_mat_iat[1, :, i_atom]
                    diff_D_fp_z = D_fp_mat_iat[2, :, i_atom]
                del_fp[0] = del_fp[0] + const_factor*np.vdot( diff_D_fp_x, diff_fp )
                del_fp[1] = del_fp[1] + const_factor*np.vdot( diff_D_fp_y, diff_fp )
                del_fp[2] = del_fp[2] + const_factor*np.vdot( diff_D_fp_z, diff_fp )
                del_fp_dist = del_fp_dist + np.absolute( np.dot(rxyz_delta[i_atom], del_fp) )
            print ("i_iter = {0:d} del_fp_dist = {1:.6e}".format(i_iter, del_fp_dist))
            # print("del_fp_dist = ", del_fp_dist)
    
    
    fp_dist = 0.0
    for ityp in range(ntyp):
        itype = ityp + 1
        for iat in range(len(rxyz_new)):
            if types[iat] == itype:
                for jat in range(len(rxyz_new)):
                    if types[jat] == itype:
                        fp_iat = \
                        fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, iat)
                        fp_jat = \
                        fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, jat)
                        fp_dist = fp_dist + fplib_GD.get_fpdist(ntyp, types, fp_iat, fp_jat)

    print("sum of del_fp_dist = {0:.6e}".format(del_fp_dist))
    print("fp_dist = ", fp_dist)
    # return fp_dist
    # return rxyz

    

if __name__ == "__main__":
    args = sys.argv
    v1 = args[1]
    test1_CG(v1)
    test2_CG(v1)
    # test3_CG(v1)
