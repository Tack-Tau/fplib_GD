import numpy as np
import sys
from scipy.optimize import linear_sum_assignment, minimize
from scipy.linalg import null_space
import rcovdata
# import numba

# @numba.jit()
def get_gom(lseg, rxyz, rcov, amp):
    # s orbital only lseg == 1
    nat = len(rxyz)    
    if lseg == 1:
        om = np.zeros((nat, nat))
        for iat in range(nat):
            for jat in range(nat):
                d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                om[iat][jat] = np.sqrt( 4.0*r*(rcov[iat]*rcov[jat]) )**3 \
                    * np.exp(-1.0*d2*r) * amp[iat] * amp[jat]
    else:
        # for both s and p orbitals
        om = np.zeros((4*nat, 4*nat))
        for iat in range(nat):
            for jat in range(nat):
                d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                om[4*iat][4*jat] = np.sqrt( 4.0*r*(rcov[iat]*rcov[jat]) )**3 \
                    * np.exp(-1.0*d2*r) * amp[iat] * amp[jat]
                
                # <s_i | p_j>
                sji = np.sqrt(4.0*rcov[iat]*rcov[jat])**3 * np.exp(-1*d2*r)
                stv = np.sqrt(8.0) * rcov[jat] * r * sji
                om[4*iat][4*jat+1] = stv * d[0] * amp[iat] * amp[jat]
                om[4*iat][4*jat+2] = stv * d[1] * amp[iat] * amp[jat]
                om[4*iat][4*jat+3] = stv * d[2] * amp[iat] * amp[jat]

                # <p_i | s_j> 
                stv = np.sqrt(8.0) * rcov[iat] * r * sji * -1.0
                om[4*iat+1][4*jat] = stv * d[0] * amp[iat] * amp[jat]
                om[4*iat+2][4*jat] = stv * d[1] * amp[iat] * amp[jat]
                om[4*iat+3][4*jat] = stv * d[2] * amp[iat] * amp[jat]

                # <p_i | p_j>
                stv = -8.0 * rcov[iat] * rcov[jat] * r * r * sji
                om[4*iat+1][4*jat+1] = stv * (d[0] * d[0] - 0.5/r) * amp[iat] * amp[jat]
                om[4*iat+1][4*jat+2] = stv * (d[1] * d[0]        ) * amp[iat] * amp[jat]
                om[4*iat+1][4*jat+3] = stv * (d[2] * d[0]        ) * amp[iat] * amp[jat]
                om[4*iat+2][4*jat+1] = stv * (d[0] * d[1]        ) * amp[iat] * amp[jat]
                om[4*iat+2][4*jat+2] = stv * (d[1] * d[1] - 0.5/r) * amp[iat] * amp[jat]
                om[4*iat+2][4*jat+3] = stv * (d[2] * d[1]        ) * amp[iat] * amp[jat]
                om[4*iat+3][4*jat+1] = stv * (d[0] * d[2]        ) * amp[iat] * amp[jat]
                om[4*iat+3][4*jat+2] = stv * (d[1] * d[2]        ) * amp[iat] * amp[jat]
                om[4*iat+3][4*jat+3] = stv * (d[2] * d[2] - 0.5/r) * amp[iat] * amp[jat]
    
    # for i in range(len(om)):
    #     for j in range(len(om)):
    #         if abs(om[i][j] - om[j][i]) > 1e-6:
    #             print ("ERROR", i, j, om[i][j], om[j][i])
    return om



# @numba.jit()
def kron_delta(i,j):
    if i == j:
        m = 1.0
    else:
        m = 0.0
    return m



# @numba.jit()
def get_D_gom(lseg, rxyz, rcov, amp, D_n):
    # s orbital only lseg == 1
    nat = len(rxyz)    
    if lseg == 1:
        D_om = np.zeros((3, nat, nat))
        for x in range(3):
            for iat in range(nat):
                for jat in range(nat):
                    d = rxyz[iat] - rxyz[jat]
                    d2 = np.vdot(d, d)
                    r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                    sji = np.sqrt( 4.0*r*(rcov[iat]*rcov[jat]) )**3 * np.exp(-1.0*d2*r)
                    # Derivative of <s_i | s_j>
                    D_om[x][iat][jat] = -( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * \
                                   (2.0*r) * d[x] * sji * amp[iat] * amp[jat]
                
    else:
        # for both s and p orbitals
        D_om = np.zeros((3, 4*nat, 4*nat))
        for x in range(3):
            for iat in range(nat):
                for jat in range(nat):
                    d = rxyz[iat] - rxyz[jat]
                    d2 = np.vdot(d, d)
                    r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                    sji = np.sqrt(4.0*rcov[iat]*rcov[jat])**3 * np.exp(-1.0*d2*r)
                    # Derivative of <s_i | s_j>
                    D_om[x][4*iat][4*jat] = -( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * \
                                   (2.0*r) * d[x] * sji * amp[iat] * amp[jat]
                
                    # Derivative of <s_i | p_j>
                    stv = np.sqrt(8.0) * rcov[jat] * r * sji
                    for i_sp in range(3):
                        D_om[x][4*iat][4*jat+i_sp+1] = \
                        ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * \
                        stv * amp[iat] * amp[jat] * ( kron_delta(x, i_sp) - \
                                                     np.dot( d[x], d[i_sp] ) * 2.0*r  )

                    # Derivative of <p_i | s_j>
                    stv = np.sqrt(8.0) * rcov[iat] * r * sji * -1.0
                    for i_ps in range(3):
                        D_om[x][4*iat+i_ps+1][4*jat] = \
                        ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * \
                        stv * amp[iat] * amp[jat] * ( kron_delta(x, i_ps) - \
                                                     np.dot( d[x], d[i_ps] ) * 2.0*r )

                    # Derivative of <p_i | p_j>
                    stv = -8.0 * rcov[iat] * rcov[jat] * r * r * sji
                    for i_pp in range(3):
                        for j_pp in range(3):
                            D_om[x][4*iat+i_pp+1][4*jat+j_pp+1] = \
                            ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * \
                            d[x] * stv * amp[iat] * amp[jat] * \
                            ( kron_delta(x, j_pp) - 2.0 * r * d[i_pp] * d[j_pp] ) + \
                            ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * \
                            stv * amp[iat] * amp[jat] * ( kron_delta(x, i_pp) * d[j_pp] + \
                                                         kron_delta(x, j_pp) * d[i_pp] )
                
    return D_om



# @numba.jit()
def get_D_fp(contract, ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, x, D_n, iat):
    if lmax == 0:
        lseg = 1
        l = 1
    else:
        lseg = 4
        l = 2
    amp, n_sphere, rxyz_sphere, rcov_sphere = \
                   get_sphere(ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, iat)
    om = get_gom(lseg, rxyz_sphere, rcov_sphere, amp)
    lamda_om, Varr_om = np.linalg.eig(om)
    lamda_om = np.real(lamda_om)
    
    # Sort eigen_val & eigen_vec joint matrix in corresponding descending order of eigen_val
    lamda_Varr_om = np.vstack((lamda_om, Varr_om))
    sorted_lamda_Varr_om = lamda_Varr_om[ :, lamda_Varr_om[0].argsort()]
    sorted_Varr_om = sorted_lamda_Varr_om[1:, :]
    
    N_vec = len(sorted_Varr_om[0])
    D_fp = np.zeros((nx*lseg, 1)) + 1j*np.zeros((nx*lseg, 1))
    # D_fp = np.zeros((nx*lseg, 1))
    D_om = get_D_gom(lseg, rxyz_sphere, rcov_sphere, amp, D_n)
    if x == 0:
        Dx_om = D_om[0, :, :]
        for i in range(N_vec):
            Dx_mul_V_om = np.matmul(Dx_om, sorted_Varr_om[:, i])
            D_fp[i][0] = np.matmul(sorted_Varr_om[:, i].T, Dx_mul_V_om)
    elif x == 1:
        Dy_om = D_om[1, :, :]
        for j in range(N_vec):
            Dy_mul_V_om = np.matmul(Dy_om, sorted_Varr_om[:, j])
            D_fp[j][0] = np.matmul(sorted_Varr_om[:, j].T, Dy_mul_V_om)
    elif x == 2:
        Dz_om = D_om[2, :, :]
        for k in range(N_vec):
            Dz_mul_V_om = np.matmul(Dz_om, sorted_Varr_om[:, k])
            D_fp[k][0] = np.matmul(sorted_Varr_om[:, k].T, Dz_mul_V_om)
    else:
        print("Error: Wrong x value! x can only be 0,1,2")
    
    # D_fp = np.real(D_fp)
    # print("D_fp {0:d} = {1:s}".format(x, np.array_str(D_fp, precision=6, suppress_small=False)) )
    # D_fp_factor = np.zeros(N_vec)
    # D_fp_factor = np.zeros(N_vec) + 1j*np.zeros(N_vec)
    # for N in range(N_vec):
    #     D_fp_factor[N] = 1/D_fp[N][0]
    #     D_fp[N][0] = (np.exp( np.log(D_fp_factor[N]*D_fp[N][0] + 1.2) ) - 1.2)/D_fp_factor[N]
    return D_fp


# @numba.jit()
def get_D_fp_mat(contract, ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, iat):
    if lmax == 0:
        lseg = 1
        l = 1
    else:
        lseg = 4
        l = 2
    # amp, n_sphere, rxyz_sphere, rcov_sphere = \
    #               get_sphere(ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, iat)
    # om = get_gom(lseg, rxyz_sphere, rcov_sphere, amp)
    # lamda_om, Varr_om = np.linalg.eig(om)
    # lamda_om = np.real(lamda_om)
    # N_vec = len(Varr_om[0])
    nat = len(rxyz)
    D_fp_mat = np.zeros((nat, 3, nx*lseg)) + 1j*np.zeros((nat, 3, nx*lseg))
    for i in range(3*nat):
        D_n = i // 3
        x = i % 3
        D_fp = get_D_fp(contract, ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, x, D_n, iat)
        for j in range(len(D_fp)):
            D_fp_mat[D_n][x][j] = D_fp[j][0]
            # D_fp_mat[x, :, D_n] = D_fp
            # Another way to compute D_fp_mat is through looping np.column_stack((a,b))
    # print("D_fp_mat = \n{0:s}".format(np.array_str(D_fp_mat, precision=6, suppress_small=False)) )
    return D_fp_mat



def get_common_sphere(ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, iat, jat):
    amp_j, n_sphere_j, rxyz_sphere_j, rcov_sphere_j = \
                get_sphere(ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, jat)
    i_sphere_count = 0
    nat_j_sphere = len(rxyz_sphere_j)
    iat_in_j_sphere = False
    rxyz_list = rxyz.tolist()
    rxyz_sphere_j_list = rxyz_sphere_j.tolist()
    for j in range(nat_j_sphere):
        if rxyz_list[iat] == rxyz_sphere_j_list[j]:
            iat_in_j_sphere = True
            return iat_in_j_sphere, j
        else:
            return iat_in_j_sphere, j



'''
def get_common_sphere(ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, iat, jat):
    amp_1, n_sphere_1, rxyz_sphere_1, rcov_sphere_1 = \
                get_sphere(ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, iat)
    amp_2, n_sphere_2, rxyz_sphere_2, rcov_sphere_2 = \
                get_sphere(ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, jat)
    nat_1 = len(rxyz_sphere_1)
    nat_2 = len(rxyz_sphere_2)
    rxyz_sphere_1 = rxyz_sphere_1.tolist()
    rxyz_sphere_2 = rxyz_sphere_2.tolist()
    i_rxyz_sphere_1 = []
    i_rxyz_sphere_2 = []
    common_count = 0
    for i in range(nat_1):
        for j in range(nat_2):
            if rxyz_sphere_1[i] == rxyz_sphere_2[j]:
                common_count = common_count + 1
                i_rxyz_sphere_1.append(i)
                i_rxyz_sphere_2.append(j)
    # print("{0:d} common atoms for {1:d}th atom and {2:d}th atom".format(common_count, iat+1, jat+1))
    return common_count, i_rxyz_sphere_1, i_rxyz_sphere_2
'''    



# Previous get_D_om
'''
# @numba.jit()
def get_Dx_gom(lseg, rxyz, rcov, amp, D_n):
    # s orbital only lseg == 1
    nat = len(rxyz)    
    if lseg == 1:
        Dx_om = np.zeros((nat, nat))
        for iat in range(nat):
            for jat in range(nat):
                d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                sji = np.sqrt( 4.0*r*(rcov[iat]*rcov[jat]) )**3 * np.exp(-1.0*d2*r)
                # Derivative of <s_i | s_j>
                Dx_om[iat][jat] = -( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * \
                                   (2.0*r) * d[0] * sji * amp[iat] * amp[jat]
                
    else:
        # for both s and p orbitals
        Dx_om = np.zeros((4*nat, 4*nat))
        for iat in range(nat):
            for jat in range(nat):
                d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                sji = np.sqrt(4.0*rcov[iat]*rcov[jat])**3 * np.exp(-1.0*d2*r)
                # Derivative of <s_i | s_j>
                Dx_om[4*iat][4*jat] = -( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * \
                                   (2.0*r) * d[0] * sji * amp[iat] * amp[jat]
                
                # Derivative of <s_i | p_j>
                stv = np.sqrt(8.0) * rcov[jat] * r * sji
                Dx_om[4*iat][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[0], d[0] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dx_om[4*iat][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * (                       0.0 - stv * np.dot( d[0], d[1] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dx_om[4*iat][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * (                       0.0 - stv * np.dot( d[0], d[2] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )

                # Derivative of <p_i | s_j>
                stv = np.sqrt(8.0) * rcov[iat] * r * sji * -1.0
                Dx_om[4*iat+1][4*jat] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[0], d[0] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dx_om[4*iat+2][4*jat] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * (                       0.0 - stv * np.dot( d[0], d[1] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dx_om[4*iat+3][4*jat] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * (                       0.0 - stv * np.dot( d[0], d[2] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )

                # Derivative of <p_i | p_j>
                stv = -8.0 * rcov[iat] * rcov[jat] * r * r * sji
                Dx_om[4*iat+1][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (1.0 - 2.0 * r * d[0] * d[0]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[0] + d[0]) \
                                                                     * amp[iat] * amp[jat]
                Dx_om[4*iat+1][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (0.0 - 2.0 * r * d[0] * d[1]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[1] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dx_om[4*iat+1][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (0.0 - 2.0 * r * d[0] * d[2]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[2] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dx_om[4*iat+2][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (0.0 - 2.0 * r * d[1] * d[0]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (0.0  + d[1]) \
                                                                     * amp[iat] * amp[jat]
                Dx_om[4*iat+2][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (1.0 - 2.0 * r * d[1] * d[1]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (0.0  +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dx_om[4*iat+2][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (0.0 - 2.0 * r * d[1] * d[2]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (0.0  +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dx_om[4*iat+3][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (0.0 - 2.0 * r * d[2] * d[0]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (0.0  + d[2]) \
                                                                     * amp[iat] * amp[jat]
                Dx_om[4*iat+3][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (0.0 - 2.0 * r * d[2] * d[1]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (0.0  +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dx_om[4*iat+3][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (1.0 - 2.0 * r * d[2] * d[2]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (0.0  +  0.0) \
                                                                     * amp[iat] * amp[jat]
                
    return Dx_om



# @numba.jit()
def get_Dy_gom(lseg, rxyz, rcov, amp, D_n):
    # s orbital only lseg == 1
    nat = len(rxyz)    
    if lseg == 1:
        Dy_om = np.zeros((nat, nat))
        for iat in range(nat):
            for jat in range(nat):
                d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                sji = np.sqrt( 4.0*r*(rcov[iat]*rcov[jat]) )**3 * np.exp(-1.0*d2*r)
                # Derivative of <s_i | s_j>
                Dy_om[iat][jat] = -( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * \
                                   (2.0*r) * d[1] * sji * amp[iat] * amp[jat]
                
    else:
        # for both s and p orbitals
        Dy_om = np.zeros((4*nat, 4*nat))
        for iat in range(nat):
            for jat in range(nat):
                d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                sji = np.sqrt(4.0*rcov[iat]*rcov[jat])**3 * np.exp(-1*d2*r)
                # Derivative of <s_i | s_j>
                Dy_om[4*iat][4*jat] = -( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * \
                                   (2.0*r) * d[1] * sji * amp[iat] * amp[jat]
                
                # Derivative of <s_i | p_j>
                stv = np.sqrt(8.0) * rcov[jat] * r * sji
                Dy_om[4*iat][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * (                       0.0 - stv * np.dot( d[1], d[0] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dy_om[4*iat][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[1], d[1] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dy_om[4*iat][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * (                       0.0 - stv * np.dot( d[1], d[2] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )

                # Derivative of <p_i | s_j>
                stv = np.sqrt(8.0) * rcov[iat] * r * sji * -1.0
                Dy_om[4*iat+1][4*jat] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * (                       0.0 - stv * np.dot( d[1], d[0] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dy_om[4*iat+2][4*jat] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[1], d[1] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dy_om[4*iat+3][4*jat] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * (                       0.0 - stv * np.dot( d[1], d[2] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )

                # Derivative of <p_i | p_j>
                stv = -8.0 * rcov[iat] * rcov[jat] * r * r * sji
                Dy_om[4*iat+1][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[1] * stv * (1.0 - 2.0 * r * d[0] * d[0]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (0.0  +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dy_om[4*iat+1][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[1] * stv * (0.0 - 2.0 * r * d[0] * d[1]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (0.0  + d[0]) \
                                                                     * amp[iat] * amp[jat]
                Dy_om[4*iat+1][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[1] * stv * (0.0 - 2.0 * r * d[0] * d[2]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (0.0  +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dy_om[4*iat+2][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[1] * stv * (0.0 - 2.0 * r * d[1] * d[0]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[0] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dy_om[4*iat+2][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[1] * stv * (1.0 - 2.0 * r * d[1] * d[1]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[1] + d[1]) \
                                                                     * amp[iat] * amp[jat]
                Dy_om[4*iat+2][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[1] * stv * (0.0 - 2.0 * r * d[1] * d[2]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[2] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dy_om[4*iat+3][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[1] * stv * (0.0 - 2.0 * r * d[2] * d[0]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (0.0  +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dy_om[4*iat+3][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[1] * stv * (0.0 - 2.0 * r * d[2] * d[1]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (0.0  + d[2]) \
                                                                     * amp[iat] * amp[jat]
                Dy_om[4*iat+3][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[1] * stv * (1.0 - 2.0 * r * d[2] * d[2]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (0.0  +  0.0) \
                                                                     * amp[iat] * amp[jat]
                
    return Dy_om



# @numba.jit()
def get_Dz_gom(lseg, rxyz, rcov, amp, D_n):
    # s orbital only lseg == 1
    nat = len(rxyz)    
    if lseg == 1:
        Dz_om = np.zeros((nat, nat))
        for iat in range(nat):
            for jat in range(nat):
                d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                sji = np.sqrt( 4.0*r*(rcov[iat]*rcov[jat]) )**3 * np.exp(-1.0*d2*r)
                # Derivative of <s_i | s_j>
                Dz_om[iat][jat] = -( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * \
                                   (2.0*r) * d[2] * sji * amp[iat] * amp[jat]
                
    else:
        # for both s and p orbitals
        Dz_om = np.zeros((4*nat, 4*nat))
        for iat in range(nat):
            for jat in range(nat):
                d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                sji = np.sqrt(4.0*rcov[iat]*rcov[jat])**3 * np.exp(-1*d2*r)
                # Derivative of <s_i | s_j>
                Dz_om[4*iat][4*jat] = -( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * \
                                   (2.0*r) * d[2] * sji * amp[iat] * amp[jat]
                
                # Derivative of <s_i | p_j>
                stv = np.sqrt(8.0) * rcov[jat] * r * sji
                Dz_om[4*iat][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * (                       0.0 - stv * np.dot( d[2], d[0] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dz_om[4*iat][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * (                       0.0 - stv * np.dot( d[2], d[1] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dz_om[4*iat][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[2], d[2] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )

                # Derivative of <p_i | s_j>
                stv = np.sqrt(8.0) * rcov[iat] * r * sji * -1.0
                Dz_om[4*iat+1][4*jat] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * (                       0.0 - stv * np.dot( d[2], d[0] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dz_om[4*iat+2][4*jat] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * (                       0.0 - stv * np.dot( d[2], d[1] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dz_om[4*iat+3][4*jat] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[2], d[2] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )

                # Derivative of <p_i | p_j>
                stv = -8.0 * rcov[iat] * rcov[jat] * r * r * sji
                Dz_om[4*iat+1][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[2] * stv * (1.0 - 2.0 * r * d[0] * d[0]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (0.0  +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dz_om[4*iat+1][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[2] * stv * (0.0 - 2.0 * r * d[0] * d[1]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (0.0  +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dz_om[4*iat+1][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[2] * stv * (0.0 - 2.0 * r * d[0] * d[2]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (0.0  + d[0]) \
                                                                     * amp[iat] * amp[jat]
                Dz_om[4*iat+2][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[2] * stv * (0.0 - 2.0 * r * d[1] * d[0]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (0.0  +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dz_om[4*iat+2][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[2] * stv * (1.0 - 2.0 * r * d[1] * d[1]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (0.0  +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dz_om[4*iat+2][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[2] * stv * (0.0 - 2.0 * r * d[1] * d[2]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (0.0  + d[1]) \
                                                                     * amp[iat] * amp[jat]
                Dz_om[4*iat+3][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[2] * stv * (0.0 - 2.0 * r * d[2] * d[0]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[0] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dz_om[4*iat+3][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[2] * stv * (0.0 - 2.0 * r * d[2] * d[1]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[1] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dz_om[4*iat+3][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[2] * stv * (1.0 - 2.0 * r * d[2] * d[2]) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[2] + d[2]) \
                                                                     * amp[iat] * amp[jat]
                
    return Dz_om

'''


#################################################################################
# Self-contained implementation of non-linear optimization algorithms:
# https://github.com/yrlu/non-convex
# https://github.com/tamland/non-linear-optimization
#################################################################################



# @numba.jit()
def get_fp_nonperiodic(rxyz, znucls):
    rcov = []
    amp = [1.0] * len(rxyz)
    for x in znucls:
        rcov.append(rcovdata.rcovdata[x][2])
    gom = get_gom(1, rxyz, rcov, amp)
    fp = np.linalg.eigvals(gom)
    fp = sorted(fp)
    fp = np.array(fp, float)
    return fp



# @numba.jit()
def get_fpdist_nonperiodic(fp1, fp2):
    d = fp1 - fp2
    return np.sqrt(np.vdot(d, d))


# @numba.jit()
def get_sphere(ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, iat):
    if lmax == 0:
        lseg = 1
        l = 1
    else:
        lseg = 4
        l = 2
    ixyz = get_ixyz(lat, cutoff)
    NC = 3
    wc = cutoff / np.sqrt(2.* NC)
    fc = 1.0 / (2.0 * NC * wc**2)
    nat = len(rxyz)
    cutoff2 = cutoff**2 
    n_sphere_list = []
    # print ("init iat = ", iat)
    # if iat > (nat-1):
        # print ("max iat = ", iat)
        # sys.exit("Error: ith atom (iat) is out of the boundary of the original unit cell (POSCAR)")
        # return amp, n_sphere, rxyz_sphere, rcov_sphere
    # else:
        # print ("else iat = ", iat)
    if iat <= (nat-1):
        rxyz_sphere = []
        rcov_sphere = []
        ind = [0] * (lseg * nx)
        amp = []
        xi, yi, zi = rxyz[iat]
        n_sphere = 0
        for jat in range(nat):
            for ix in range(-ixyz, ixyz+1):
                for iy in range(-ixyz, ixyz+1):
                    for iz in range(-ixyz, ixyz+1):
                        xj = rxyz[jat][0] + ix*lat[0][0] + iy*lat[1][0] + iz*lat[2][0]
                        yj = rxyz[jat][1] + ix*lat[0][1] + iy*lat[1][1] + iz*lat[2][1]
                        zj = rxyz[jat][2] + ix*lat[0][2] + iy*lat[1][2] + iz*lat[2][2]
                        d2 = (xj-xi)**2 + (yj-yi)**2 + (zj-zi)**2
                        if d2 <= cutoff2:
                            n_sphere += 1
                            if n_sphere > nx:
                                print ("FP WARNING: the cutoff is too large.")
                            amp.append((1.0-d2*fc)**NC)
                            # print (1.0-d2*fc)**NC
                            rxyz_sphere.append([xj, yj, zj])
                            rcov_sphere.append(rcovdata.rcovdata[znucl[types[jat]-1]][2]) 
                            if jat == iat and ix == 0 and iy == 0 and iz == 0:
                                ityp_sphere = 0
                            else:
                                ityp_sphere = types[jat]
                            '''
                            for il in range(lseg):
                                if il == 0:
                                    # print len(ind)
                                    # print ind
                                    # print il+lseg*(n_sphere-1)
                                    ind[il+lseg*(n_sphere-1)] = ityp_sphere * l
                                else:
                                    ind[il+lseg*(n_sphere-1)] = ityp_sphere * l + 1
                                    # ind[il+lseg*(n_sphere-1)] == ityp_sphere * l + 1
                             '''
        n_sphere_list.append(n_sphere)
        rxyz_sphere = np.array(rxyz_sphere, float)
    # for n_iter in range(nx-n_sphere+1):
        # rxyz_sphere.append([0.0, 0.0, 0.0])
        # rxyz_sphere.append([0.0, 0.0, 0.0])
    # rxyz_sphere = np.array(rxyz_sphere, float)
    # print ("amp", amp)
    # print ("n_sphere", n_sphere)
    # print ("rxyz_sphere", rxyz_sphere)
    # print ("rcov_sphere", rcov_sphere)
    return amp, n_sphere, rxyz_sphere, rcov_sphere



# @numba.jit()
def get_fp(contract, ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, iat):
    if lmax == 0:
        lseg = 1
        l = 1
    else:
        lseg = 4
        l = 2
    # lfp = []
    sfp = []
    amp, n_sphere, rxyz_sphere, rcov_sphere = \
                   get_sphere(ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, iat)
    # full overlap matrix
    nid = lseg * n_sphere
    gom = get_gom(lseg, rxyz_sphere, rcov_sphere, amp)
    val, vec = np.linalg.eig(gom)
    val = np.real(val)
    # fp0 = np.zeros(nx*lseg)
    fp0 = np.zeros((nx*lseg, 1))
    for i in range(len(val)):
        # fp0[i] = val[i]
        fp0[i][0] = val[i]
    # lfp = sorted(fp0)
    lfp = fp0[ fp0[ : , 0].argsort(), : ]
    # lfp.append(sorted(fp0))
    pvec = np.real(np.transpose(vec)[0])
    # contracted overlap matrix
    if contract:
        nids = l * (ntyp + 1)
        omx = np.zeros((nids, nids))
        for i in range(nid):
            for j in range(nid):
                # print ind[i], ind[j]
                omx[ind[i]][ind[j]] = omx[ind[i]][ind[j]] + pvec[i] * gom[i][j] * pvec[j]
        # for i in range(nids):
        #     for j in range(nids):
        #         if abs(omx[i][j] - omx[j][i]) > 1e-6:
        #             print ("ERROR", i, j, omx[i][j], omx[j][i])
        # print omx
        # sfp0 = np.linalg.eigvals(omx)
        # sfp.append(sorted(sfp0))
        sfp = np.linalg.eigvals(omx)
        sfp.append(sorted(sfp))

    # print ("n_sphere_min", min(n_sphere_list))
    # print ("n_shpere_max", max(n_sphere_list)) 

    if contract:
        # sfp = np.array(sfp, float)
        sfp = np.vstack( (np.array(sfp, float), ) ).T
        return sfp
    else:
        lfp = np.array(lfp, float)
        return lfp

# @numba.jit()
def get_ixyz(lat, cutoff):
    lat2 = np.matmul(lat, np.transpose(lat))
    # print lat2
    val = np.linalg.eigvals(lat2)
    # print (vec)
    ixyz = int(np.sqrt(1.0/max(val))*cutoff) + 1
    return ixyz

# @numba.jit()
def readvasp(vp):
    buff = []
    with open(vp) as f:
        for line in f:
            buff.append(line.split())

    lat = np.array(buff[2:5], float) 
    try:
        typt = np.array(buff[5], int)
    except:
        del(buff[5])
        typt = np.array(buff[5], int)
    nat = sum(typt)
    pos = np.array(buff[7:7 + nat], float)
    types = []
    for i in range(len(typt)):
        types += [i+1]*typt[i]
    types = np.array(types, int)
    rxyz = np.dot(pos, lat)
    # rxyz = pos
    return lat, rxyz, types



# @numba.jit()
def get_rxyz_delta(rxyz):
    nat = len(rxyz)
    rxyz_delta = np.subtract( np.random.rand(nat, 3), 0.5*np.ones((nat, 3)) )
    for iat in range(nat):
        r_norm = np.linalg.norm(rxyz_delta[iat])
        rxyz_delta[iat] = np.divide(rxyz_delta[iat], r_norm)
    # rxyz_plus = np.add(rxyz, rxyz_delta)
    # rxyz_minus = np.subtract(rxyz, rxyz_delta)
        
    return rxyz_delta



# @numba.jit()
def get_fpdist(ntyp, types, fp1, fp2):
    # lenfp, = np.shape(fp1)
    # nat, lenfp = np.shape(fp1)
    # fpd = 0.0
    tfpd = fp1 - fp2
    # fpd = np.sqrt( np.dot(tfpd, tfpd)/lenfp )
    # fpd = np.dot(tfpd, tfpd)/lenfp
    fpd = np.matmul(tfpd.T, tfpd)
    return fpd

# Calculate crystal atomic finger print energy
def get_fp_energy(v1):
    ntyp = 1
    nx = 300
    lmax = 0
    cutoff = 6.5
    znucl = np.array([3], int)
    lat, rxyz, types = readvasp(v1)
    contract = False
    fp_dist = 0.0
    fpdist_error = 0.0
    temp_num = 0.0
    temp_sum = 0.0
    for ityp in range(ntyp):
        itype = ityp + 1
        for i_atom in range(len(rxyz)):
            if types[i_atom] == itype:
                for j_atom in range(len(rxyz)):
                    if types[j_atom] == itype:
                        fp_iat = \
                        get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff, i_atom)
                        fp_jat = \
                        get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff, j_atom)
                        temp_num = fpdist_error + get_fpdist(ntyp, types, fp_iat, fp_jat)
                        temp_sum = fp_dist + temp_num
                        accum_error = temp_num - (temp_sum - fp_dist)
                        fp_dist = temp_sum

    
    # print ( "Finger print energy = {0:s}".format(np.array_str(fp_dist, \
    #                                            precision=6, suppress_small=False)) )
    return fp_dist

#Calculate crystal atomic finger print force and steepest descent update
def get_fp_forces(v1):
    ntyp = 1
    nx = 300
    lmax = 0
    cutoff = 6.5
    znucl = np.array([3], int)
    lat, rxyz, types = fplib_GD.readvasp(v1)
    contract = False
    i_iter = 0
    iter_max = 20
    atol = 1e-6
    step_size = 1e-4
    rxyz_new = rxyz.copy()
    # fp_dist = 0.0
    # fpdist_error = 0.0
    # fpdist_temp_sum = 0.0
    # fpdsit_temp_num = 0.0
    
    for i_iter in range(iter_max+1):
        del_fp = np.zeros((len(rxyz_new), 3))
        sum_del_fp = np.zeros(3)
        fp_dist = 0.0
        for i_atom in range(len(rxyz_new)):
            # del_fp = np.zeros(3)
            # temp_del_fp = np.zeros(3)
            # accum_error = np.zeros(3)
            # temp_sum = np.zeros(3)
            for j_atom in range(len(rxyz_new)):
                fp_iat = \
                get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, i_atom)
                fp_jat = \
                get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, j_atom)
                D_fp_mat_iat = \
                get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, i_atom)
                D_fp_mat_jat = \
                get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, j_atom)
                diff_fp = fp_iat-fp_jat
                
                iat_in_j_sphere, iat_j = get_common_sphere(ntyp, \
                              nx, lmax, lat, rxyz_new, types, znucl, cutoff, i_atom, j_atom)
                if iat_in_j_sphere:
                    diff_D_fp_x = D_fp_mat_iat[i_atom, 0, :] - D_fp_mat_jat[iat_j, 0, :]
                    diff_D_fp_y = D_fp_mat_iat[i_atom, 1, :] - D_fp_mat_jat[iat_j, 1, :]
                    diff_D_fp_z = D_fp_mat_iat[i_atom, 2, :] - D_fp_mat_jat[iat_j, 2, :]
                else:
                    diff_D_fp_x = D_fp_mat_iat[i_atom, 0, :]
                    diff_D_fp_y = D_fp_mat_iat[i_atom, 1, :]
                    diff_D_fp_z = D_fp_mat_iat[i_atom, 2, :]
                
                # Kahan sum implementation
                '''
                diff_D_fp_x = np.vstack( (np.array(diff_D_fp_x)[::-1], ) ).T
                diff_D_fp_y = np.vstack( (np.array(diff_D_fp_y)[::-1], ) ).T
                diff_D_fp_z = np.vstack( (np.array(diff_D_fp_z)[::-1], ) ).T
                temp_del_fp[0] = accum_error[0] + np.real( np.matmul( diff_fp.T,  diff_D_fp_x ) )
                temp_del_fp[1] = accum_error[1] + np.real( np.matmul( diff_fp.T,  diff_D_fp_y ) )
                temp_del_fp[2] = accum_error[2] + np.real( np.matmul( diff_fp.T,  diff_D_fp_z ) )
                temp_sum[0] = del_fp[0] + temp_del_fp[0]
                temp_sum[1] = del_fp[1] + temp_del_fp[1]
                temp_sum[2] = del_fp[2] + temp_del_fp[2]
                accum_error[0] = temp_del_fp[0] - (temp_sum[0] - del_fp[0])
                accum_error[1] = temp_del_fp[1] - (temp_sum[1] - del_fp[1])
                accum_error[2] = temp_del_fp[2] - (temp_sum[2] - del_fp[2])
                del_fp[0] = temp_sum[0]
                del_fp[1] = temp_sum[1]
                del_fp[2] = temp_sum[2]
                
                fpdist_temp_num = fpdist_error + fplib_GD.get_fpdist(ntyp, types, fp_iat, fp_jat)
                fpdist_temp_sum = fp_dist + fpdist_temp_num
                fpdist_error = fpdist_temp_num - (fpdist_temp_sum - fp_dist)
                fp_dist = fpdist_temp_sum
                '''
                
                
                diff_D_fp_x = np.vstack( (np.array(diff_D_fp_x)[::-1], ) ).T
                diff_D_fp_y = np.vstack( (np.array(diff_D_fp_y)[::-1], ) ).T
                diff_D_fp_z = np.vstack( (np.array(diff_D_fp_z)[::-1], ) ).T
                del_fp[i_atom][0] = del_fp[i_atom][0] + \
                                    2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_x ) )
                del_fp[i_atom][1] = del_fp[i_atom][1] + \
                                    2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_y ) )
                del_fp[i_atom][2] = del_fp[i_atom][2] + \
                                    2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_z ) )
                fp_dist = fp_dist + fplib_GD.get_fpdist(ntyp, types, fp_iat, fp_jat)
                
                # Debugging
                '''
                print ("diff_D_fp_x = \n{0:s}".\
                      format(np.array_str(diff_D_fp_x, precision=6, suppress_small=False)))
                print ("diff_D_fp_y = \n{0:s}".\
                      format(np.array_str(diff_D_fp_y, precision=6, suppress_small=False)))
                print ("diff_D_fp_z = \n{0:s}".\
                      format(np.array_str(diff_D_fp_x, precision=6, suppress_small=False), \
                             np.array_str(diff_D_fp_y, precision=6, suppress_small=False), \
                             np.array_str(diff_D_fp_z, precision=6, suppress_small=False)))
                
                print ( "diff_fp = \n{0:s}".\
                      format(np.array_str(diff_fp, precision=6, suppress_small=False)) )
                print ( "del_fp = [{0:.6e}, {1:.6e}, {2:.6e}]".\
                      format(del_fp[i_atom][0], del_fp[i_atom][1], del_fp[i_atom][2]) )
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
            
            # rxyz_new[i_atom] = rxyz_new[i_atom] - step_size*del_fp/np.linalg.norm(del_fp)
            
        sum_del_fp = np.sum(del_fp, axis=0)
        for ii_atom in range(len(rxyz_new)):
            del_fp[ii_atom, :] = del_fp[ii_atom, :] - sum_del_fp/len(rxyz_new)
            rxyz_new[ii_atom] = rxyz_new[ii_atom] - \
                                step_size*del_fp[ii_atom, :]/np.linalg.norm(del_fp[ii_atom, :])
        print ( "i_iter = {0:d} \nrxyz_final = \n{1:s}".\
              format(i_iter+1, np.array_str(rxyz_new, precision=6, suppress_small=False)) )
        print ( "Forces = \n{0:s}".\
              format(np.array_str(del_fp, precision=6, suppress_small=False)) )
        print ( "Finger print energy difference = {0:s}".\
              format(np.array_str(fp_dist, precision=6, suppress_small=False)) )
            
    
    return del_fp
