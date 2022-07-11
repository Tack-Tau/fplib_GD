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
        mamp = np.zeros((nat, nat))
        for iat in range(nat):
            for jat in range(nat):
                d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                om[iat][jat] = np.sqrt( 4.0*r*(rcov[iat]*rcov[jat]) )**3 \
                    * np.exp(-1.0*d2*r)
                mamp[iat][jat] = amp[iat] * amp[jat]
    else:
        # for both s and p orbitals
        om = np.zeros((4*nat, 4*nat))
        mamp = np.zeros((4*nat, 4*nat))
        for iat in range(nat):
            for jat in range(nat):
                d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                om[4*iat][4*jat] = np.sqrt( 4.0*r*(rcov[iat]*rcov[jat]) )**3 \
                    * np.exp(-1.0*d2*r)
                mamp[iat][jat] = amp[iat] * amp[jat]
                
                # <s_i | p_j>
                sji = np.sqrt(4.0*r*rcov[iat]*rcov[jat])**3 * np.exp(-1*d2*r)
                stv = np.sqrt(8.0) * rcov[jat] * r * sji
                om[4*iat][4*jat+1] = stv * d[0]
                om[4*iat][4*jat+2] = stv * d[1]
                om[4*iat][4*jat+3] = stv * d[2]
                                
                mamp[4*iat][4*jat+1] = amp[iat] * amp[jat]
                mamp[4*iat][4*jat+2] = amp[iat] * amp[jat]
                mamp[4*iat][4*jat+3] = amp[iat] * amp[jat]

                # <p_i | s_j> 
                stv = np.sqrt(8.0) * rcov[iat] * r * sji * -1.0
                om[4*iat+1][4*jat] = stv * d[0]
                om[4*iat+2][4*jat] = stv * d[1]
                om[4*iat+3][4*jat] = stv * d[2]
                
                mamp[4*iat+1][4*jat] = amp[iat] * amp[jat]
                mamp[4*iat+2][4*jat] = amp[iat] * amp[jat]
                mamp[4*iat+3][4*jat] = amp[iat] * amp[jat]

                # <p_i | p_j>
                stv = -8.0 * rcov[iat] * rcov[jat] * r * r * sji
                om[4*iat+1][4*jat+1] = stv * (d[0] * d[0] - 0.5/r)
                om[4*iat+1][4*jat+2] = stv * (d[1] * d[0]        )
                om[4*iat+1][4*jat+3] = stv * (d[2] * d[0]        )
                om[4*iat+2][4*jat+1] = stv * (d[0] * d[1]        )
                om[4*iat+2][4*jat+2] = stv * (d[1] * d[1] - 0.5/r)
                om[4*iat+2][4*jat+3] = stv * (d[2] * d[1]        )
                om[4*iat+3][4*jat+1] = stv * (d[0] * d[2]        )
                om[4*iat+3][4*jat+2] = stv * (d[1] * d[2]        )
                om[4*iat+3][4*jat+3] = stv * (d[2] * d[2] - 0.5/r)
                
                mamp[4*iat+1][4*jat+1] = amp[iat] * amp[jat]
                mamp[4*iat+1][4*jat+2] = amp[iat] * amp[jat]
                mamp[4*iat+1][4*jat+3] = amp[iat] * amp[jat]
                mamp[4*iat+2][4*jat+1] = amp[iat] * amp[jat]
                mamp[4*iat+2][4*jat+2] = amp[iat] * amp[jat]
                mamp[4*iat+2][4*jat+3] = amp[iat] * amp[jat]
                mamp[4*iat+3][4*jat+1] = amp[iat] * amp[jat]
                mamp[4*iat+3][4*jat+2] = amp[iat] * amp[jat]
                mamp[4*iat+3][4*jat+3] = amp[iat] * amp[jat]
    
    # for i in range(len(om)):
    #     for j in range(len(om)):
    #         if abs(om[i][j] - om[j][i]) > 1e-6:
    #             print ("ERROR", i, j, om[i][j], om[j][i])
    if check_symmetric(om) and check_pos_def(om):
        return om, mamp
    else:
        raise Exception("Gaussian Overlap Matrix is not symmetric and positive definite!")



# @numba.jit()
def check_symmetric(A, rtol = 1e-05, atol = 1e-08):
    return np.allclose(A, A.T, rtol = rtol, atol = atol)



# @numba.jit()
def check_pos_def(A):
    eps = np.finfo(float).eps
    B = A + eps*np.identity(len(A))
    if np.array_equal(B, B.T):
        try:
            np.linalg.cholesky(B)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False



# @numba.jit()
def kron_delta(i,j):
    if i == j:
        m = 1.0
    else:
        m = 0.0
    return m



# @numba.jit()
def get_D_gom(lseg, rxyz, rcov, amp, cutoff, D_n, icenter):
    # s orbital only lseg == 1
    NC = 3
    wc = cutoff / np.sqrt(2.* NC)
    fc = 1.0 / (2.0 * NC * wc**2)
    nat = len(rxyz)    
    if lseg == 1:
        D_om = np.zeros((3, nat, nat))
        for x in range(3):
            for iat in range(nat):
                for jat in range(nat):
                    d = rxyz[iat] - rxyz[jat]
                    dnc = rxyz[D_n] - rxyz[icenter]
                    d2 = np.vdot(d, d)
                    dnc2 = np.vdot(dnc, dnc)
                    r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                    sji = np.sqrt( 4.0*r*(rcov[iat]*rcov[jat]) )**3 * np.exp(-1.0*d2*r)
                    # Derivative of <s_i | s_j>
                    D_om[x][iat][jat] = -( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * \
                                   (2.0*r) * d[x] * sji * amp[iat] * amp[jat]              \
                                   -2.0 * NC * fc * dnc[x] * (1.0 - dnc2 * fc)**(NC - 1) * sji \
                                   * amp[iat] * amp[jat] * ( kron_delta(iat, D_n) - kron_delta(jat, D_n) )
                
    else:
        # for both s and p orbitals
        D_om = np.zeros((3, 4*nat, 4*nat))
        for x in range(3):
            for iat in range(nat):
                for jat in range(nat):
                    d = rxyz[iat] - rxyz[jat]
                    dnc = rxyz[D_n] - rxyz[icenter]
                    d2 = np.vdot(d, d)
                    dnc2 = np.vdot(dnc, dnc)
                    r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                    sji = np.sqrt(4.0*r*rcov[iat]*rcov[jat])**3 * np.exp(-1.0*d2*r)
                    # Derivative of <s_i | s_j>
                    D_om[x][4*iat][4*jat] = -( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * \
                                   (2.0*r) * d[x] * sji * amp[iat] * amp[jat]                  \
                                   -2.0 * NC * fc * dnc[x] * (1.0 - dnc2 * fc)**(NC - 1) * sji     \
                                   * amp[iat] * amp[jat] * ( kron_delta(iat, D_n) - kron_delta(jat, D_n) )
                
                    # Derivative of <s_i | p_j>
                    stv = np.sqrt(8.0) * rcov[jat] * r * sji
                    for i_sp in range(3):
                        D_om[x][4*iat][4*jat+i_sp+1] = \
                        ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * \
                        stv * amp[iat] * amp[jat] * ( kron_delta(x, i_sp) - \
                                                     np.dot( d[x], d[i_sp] ) * 2.0*r ) \
                        -2.0 * NC * fc * dnc[x] * (1.0 - dnc2 * fc)**(NC - 1) \
                        * stv * d[i_sp] * amp[iat] * amp[jat] * ( kron_delta(iat, D_n) - kron_delta(jat, D_n) )

                    # Derivative of <p_i | s_j>
                    stv = np.sqrt(8.0) * rcov[iat] * r * sji * -1.0
                    for i_ps in range(3):
                        D_om[x][4*iat+i_ps+1][4*jat] = \
                        ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * \
                        stv * amp[iat] * amp[jat] * ( kron_delta(x, i_ps) - \
                                                     np.dot( d[x], d[i_ps] ) * 2.0*r ) \
                        -2.0 * NC * fc * dnc[x] * (1.0 - dnc2 * fc)**(NC - 1) \
                        * stv * d[i_ps] * amp[iat] * amp[jat] * ( kron_delta(iat, D_n) - kron_delta(jat, D_n) )

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
                                                         kron_delta(x, j_pp) * d[i_pp] )  \
                            -2.0 * NC * fc * dnc[x] * (1.0 - dnc2 * fc)**(NC - 1) * stv * \
                            ( np.dot(d[i_pp], d[j_pp]) - kron_delta(i_pp, j_pp) * 0.5/r ) \
                            * amp[iat] * amp[jat] * ( kron_delta(iat, D_n) - kron_delta(jat, D_n) )
                
    return D_om



# @numba.jit()
def get_D_fp(contract, ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, x, D_n, iat):
    if lmax == 0:
        lseg = 1
        l = 1
    else:
        lseg = 4
        l = 2
    amp, sphere_id_list, icenter, rxyz_sphere, rcov_sphere = \
                   get_sphere(ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, iat)
    om, mamp = get_gom(lseg, rxyz_sphere, rcov_sphere, amp)
    gom = om * mamp
    lamda_gom, Varr_gom = np.linalg.eig(gom)
    lamda_gom = np.real(lamda_gom)
    # Adding null vectors to eigenvector matrix Varr_gom corresponding to zero eigenvalues in fp
    lamda_gom_list = lamda_gom.tolist()
    null_Varr = np.vstack( (np.zeros_like(Varr_gom[:, 0]), ) ).T
    for n in range(nx*lseg - len(lamda_gom_list)):
        lamda_gom_list.append(0.0)
        Varr_gom_new = np.hstack((Varr_gom, null_Varr))
        Varr_gom = Varr_gom_new.copy()
        
    lamda_gom = np.array(lamda_gom_list, float)
    
    # Sort eigen_val & eigen_vec joint matrix in corresponding descending order of eigen_val
    lamda_Varr_gom = np.vstack((lamda_gom, Varr_gom))
    # sorted_lamda_Varr_om = lamda_Varr_gom[ :, lamda_Varr_gom[0].argsort()]
    sorted_lamda_Varr_gom = lamda_Varr_gom[ :, lamda_Varr_gom[0].argsort()[::-1]]
    sorted_Varr_gom = sorted_lamda_Varr_gom[1:, :]
    
    N_vec = len(sorted_Varr_gom[0])
    D_fp = np.zeros((nx*lseg, 1)) + 1j*np.zeros((nx*lseg, 1))
    # D_fp = np.zeros((nx*lseg, 1))
    D_gom = get_D_gom(lseg, rxyz_sphere, rcov_sphere, amp, cutoff, D_n, icenter)
    if x == 0:
        Dx_gom = D_gom[0, :, :].copy()
        for i in range(N_vec):
            Dx_mul_V_gom = np.matmul(Dx_gom, sorted_Varr_gom[:, i])
            D_fp[i][0] = np.matmul(sorted_Varr_gom[:, i].T, Dx_mul_V_gom)
    elif x == 1:
        Dy_gom = D_gom[1, :, :].copy()
        for j in range(N_vec):
            Dy_mul_V_gom = np.matmul(Dy_gom, sorted_Varr_gom[:, j])
            D_fp[j][0] = np.matmul(sorted_Varr_gom[:, j].T, Dy_mul_V_gom)
    elif x == 2:
        Dz_gom = D_gom[2, :, :].copy()
        for k in range(N_vec):
            Dz_mul_V_gom = np.matmul(Dz_gom, sorted_Varr_gom[:, k])
            D_fp[k][0] = np.matmul(sorted_Varr_gom[:, k].T, Dz_mul_V_gom)
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
    amp, sphere_id_list, icenter, rxyz_sphere, rcov_sphere = \
                  get_sphere(ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, iat)
    # om, mamp = get_gom(lseg, rxyz_sphere, rcov_sphere, amp)
    # gom = om * mamp
    # lamda_gom, Varr_gom = np.linalg.eig(gom)
    # lamda_gom = np.real(lamda_gom)
    # N_vec = len(Varr_gom[0])
    nat = len(rxyz_sphere)
    D_fp_mat = np.zeros((3, nx*lseg, nat)) + 1j*np.zeros((3, nx*lseg, nat))
    for i in range(nat):
        if sphere_id_list[i][0:3] == [0, 0, 0]:
            # D_n = sphere_id_list[i][3]
            D_n = i
            for x in range(3):
                D_fp = get_D_fp(contract, ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, \
                                x, D_n, iat)
                for j in range(len(D_fp)):
                    D_fp_mat[x][j][D_n] = D_fp[j][0]
            # D_fp_mat[x, :, D_n] = D_fp
            # Another way to compute D_fp_mat is through looping np.column_stack((a,b))
    # print("D_fp_mat = \n{0:s}".format(np.array_str(D_fp_mat, precision=6, suppress_small=False)) )
    return D_fp_mat



def get_common_sphere(ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, iat, jat):
    amp_j, sphere_id_list_j, icenter_j, rxyz_sphere_j, rcov_sphere_j = \
                get_sphere(ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, jat)
    nat_j_sphere = len(rxyz_sphere_j)
    iat_in_j_sphere = False
    iat_j = 0
    for j in range(nat_j_sphere):
        if sphere_id_list_j[j] == [0, 0, 0, iat]:
            iat_in_j_sphere = True
            iat_j = j
            break
    return iat_in_j_sphere, iat_j



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
    om, mamp = get_gom(1, rxyz, rcov, amp)
    gom = om * mamp
    fp = np.linalg.eigvalsh(gom)
    fp = sorted(fp, reverse = True)
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
    # n_sphere_list = []
    # print ("init iat = ", iat)
    if iat > (nat-1):
        print ("max iat = ", iat)
        sys.exit("Error: ith atom (iat) is out of the boundary of the original unit cell (POSCAR)")
    else:
        rxyz_sphere = []
        rcov_sphere = []
        sphere_id_list = []
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
                                sys.exit("FP WARNING: the cutoff is too large.")
                            amp.append((1.0-d2*fc)**NC)
                            # print (1.0-d2*fc)**NC
                            rxyz_sphere.append([xj, yj, zj])
                            rcov_sphere.append(rcovdata.rcovdata[znucl[types[jat]-1]][2])
                            sphere_id_list.append([ix, iy, iz, jat])
                            if [ix, iy, iz] == [0, 0, 0] and jat == iat:
                                ityp_sphere = 0
                                icenter = n_sphere - 1
                            else:
                                ityp_sphere = types[jat]
                            
                            for il in range(lseg):
                                if il == 0:
                                    # print len(ind)
                                    # print ind
                                    # print il+lseg*(n_sphere-1)
                                    ind[il+lseg*(n_sphere-1)] = ityp_sphere * l
                                else:
                                    ind[il+lseg*(n_sphere-1)] = ityp_sphere * l + 1
                                    # ind[il+lseg*(n_sphere-1)] == ityp_sphere * l + 1
                            
        # n_sphere_list.append(n_sphere)
        rxyz_sphere = np.array(rxyz_sphere, float)
    # for n_iter in range(nx-n_sphere+1):
        # rxyz_sphere.append([0.0, 0.0, 0.0])
        # rxyz_sphere.append([0.0, 0.0, 0.0])
    # rxyz_sphere = np.array(rxyz_sphere, float)
    # print ("amp", amp)
    # print ("n_sphere", n_sphere)
    # print ("rxyz_sphere", rxyz_sphere)
    # print ("rcov_sphere", rcov_sphere)
    return amp, sphere_id_list, icenter, rxyz_sphere, rcov_sphere



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
    amp, sphere_id_list, icenter, rxyz_sphere, rcov_sphere = \
                   get_sphere(ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, iat)
    # full overlap matrix
    n_sphere = len(rxyz_sphere)
    nid = lseg * n_sphere
    om, mamp = get_gom(lseg, rxyz_sphere, rcov_sphere, amp)
    gom = om * mamp
    vals, vecs = np.linalg.eigh(gom)
    # vals = np.real(vals)
    # fp0 = np.zeros(nx*lseg)
    fp0 = np.zeros((nx*lseg, 1))
    for i in range(len(vals)):
        # fp0[i] = val[i]
        fp0[i][0] = vals[i]
    fp0_norm = np.linalg.norm(fp0)
    fp0 = np.divide(fp0, fp0_norm)
    # lfp = sorted(fp0, reverse = True)
    # lfp = fp0[ fp0[ : , 0].argsort(), : ]
    lfp = fp0[ fp0[ : , 0].argsort()[::-1], : ]
    # lfp.append(sorted(fp0, reverse = True))
    # pvec = np.real(np.transpose(vec)[0])
    vectmp = np.transpose(vecs)
    pvecs = []
    for i in range(len(vectmp)):
        pvecs.append(vectmp[len(vectmp)-1-i])
    
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
        # sfp0 = np.linalg.eigvalsh(omx)
        # sfp.append(sorted(sfp0, reverse = True))
        sfp = np.linalg.eigvalsh(omx)
        sfp.append(sorted(sfp, reverse = True))

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
    val = np.linalg.eigvalsh(lat2)
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
def read_types(vp):
    buff = []
    with open(vp) as f:
        for line in f:
            buff.append(line.split())
    try:
        typt = np.array(buff[5], int)
    except:
        del(buff[5])
        typt = np.array(buff[5], int)
    types = []
    for i in range(len(typt)):
        types += [i+1]*typt[i]
    types = np.array(types, int)
    return types



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
def get_fpdist(ntyp, types, fp1, fp2, mx = False):
    nat, lenfp = np.shape(fp1)
    fpd = 0.0
    for ityp in range(ntyp):
        itype = ityp + 1
        MX = np.zeros((nat, nat))
        for iat in range(nat):
            if types[iat] == itype:
                for jat in range(nat):
                    if types[jat] == itype:
                        tfpd = fp1[iat] - fp2[jat]
                        MX[iat][jat] = np.sqrt(np.vdot(tfpd, tfpd))

        row_ind, col_ind = linear_sum_assignment(MX)
        # print(row_ind, col_ind)
        total = MX[row_ind, col_ind].sum()
        fpd += total

    fpd = fpd / nat
    if mx:
        return fpd, col_ind
    else:
        return fpd

# Calculate crystal atomic finger print energy
def get_fp_energy(lat, rxyz, types, contract = False, ntyp = 1, nx = 300, \
                  lmax = 0, znucl = np.array([3], int), cutoff = 6.5):
    '''
    ntyp = 1
    nx = 300
    lmax = 0
    cutoff = 6.5
    znucl = np.array([3], int)
    lat, rxyz, types = readvasp(v1)
    contract = False
    '''
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
                        dfp_ij = fp_iat - fp_jat
                        temp_num = fpdist_error + np.matmul(dfp_ij.T, dfp_ij)
                        temp_sum = fp_dist + temp_num
                        accum_error = temp_num - (temp_sum - fp_dist)
                        fp_dist = temp_sum

    
    # print ( "Finger print energy = {0:s}".format(np.array_str(fp_dist, \
    #                                            precision=6, suppress_small=False)) )
    return fp_dist

#Calculate crystal atomic finger print force and steepest descent update
def get_fp_forces(lat, rxyz, types, contract = False, ntyp = 1, nx = 300, \
                  lmax = 0, znucl = np.array([3], int), cutoff = 6.5, \
                  iter_max = 1, step_size = 1e-4):
    '''
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
    '''
    rxyz_new = rxyz.copy()
    # fp_dist = 0.0
    # fpdist_error = 0.0
    # fpdist_temp_sum = 0.0
    # fpdsit_temp_num = 0.0
    
    for i_iter in range(iter_max):
        del_fp = np.zeros((len(rxyz_new), 3))
        sum_del_fp = np.zeros(3)
        # fp_dist = 0.0
        for k_atom in range(len(rxyz_new)):
            
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
                    
                    kat_in_i_sphere, kat_i = get_common_sphere(ntyp, \
                                  nx, lmax, lat, rxyz_new, types, znucl, cutoff, k_atom, i_atom)
                    kat_in_j_sphere, kat_j = get_common_sphere(ntyp, \
                                  nx, lmax, lat, rxyz_new, types, znucl, cutoff, k_atom, j_atom)
                    
                    # print("kat_in_i_sphere=", kat_in_i_sphere, "kat_i=", kat_i)
                    # print("kat_in_j_sphere=", kat_in_j_sphere, "kat_j=", kat_j)
                    # print("fp_shape=", fp_iat.shape)
                    # print("D_fp_mat_shape=", D_fp_mat_iat.shape)
                    
                    if kat_in_i_sphere == True and kat_in_j_sphere == True:
                        diff_D_fp_x = D_fp_mat_iat[0, :, kat_i] - D_fp_mat_jat[0, :, kat_j]
                        diff_D_fp_y = D_fp_mat_iat[1, :, kat_i] - D_fp_mat_jat[1, :, kat_j]
                        diff_D_fp_z = D_fp_mat_iat[2, :, kat_i] - D_fp_mat_jat[2, :, kat_j]
                    elif kat_in_i_sphere == True and kat_in_j_sphere == False:
                        diff_D_fp_x = D_fp_mat_iat[0, :, kat_i]
                        diff_D_fp_y = D_fp_mat_iat[1, :, kat_i]
                        diff_D_fp_z = D_fp_mat_iat[2, :, kat_i]
                    elif kat_in_i_sphere == False and kat_in_j_sphere == True:
                        diff_D_fp_x = - D_fp_mat_jat[0, :, kat_j]
                        diff_D_fp_y = - D_fp_mat_jat[1, :, kat_j]
                        diff_D_fp_z = - D_fp_mat_jat[2, :, kat_j]
                    else:
                        diff_D_fp_x = np.zeros_like(D_fp_mat_jat[0, :, kat_j])
                        diff_D_fp_y = np.zeros_like(D_fp_mat_jat[1, :, kat_j])
                        diff_D_fp_z = np.zeros_like(D_fp_mat_jat[2, :, kat_j])

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


                    diff_D_fp_x = np.vstack( (np.array(diff_D_fp_x), ) ).T
                    diff_D_fp_y = np.vstack( (np.array(diff_D_fp_y), ) ).T
                    diff_D_fp_z = np.vstack( (np.array(diff_D_fp_z), ) ).T
                    # print("fp_dim", fp_iat.shape)
                    # print("diff_D_fp_x_dim", diff_D_fp_x.shape)
                    
                    del_fp[i_atom][0] = del_fp[i_atom][0] + \
                                        2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_x ) )
                    del_fp[i_atom][1] = del_fp[i_atom][1] + \
                                        2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_y ) )
                    del_fp[i_atom][2] = del_fp[i_atom][2] + \
                                        2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_z ) )
                    # fp_dist = fp_dist + get_fpdist(ntyp, types, fp_iat, fp_jat)

                    # print("del_fp = ", del_fp)
                    # rxyz[i_atom] = rxyz[i_atom] - step_size*del_fp
                    '''
                    if max(del_fp) < atol:
                        print ("i_iter = {0:d} \nrxyz_final = \n{1:s}".\
                              format(i_iter+1, np.array_str(rxyz, precision=6, \
                              suppress_small=False)))
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
            '''
            print ( "i_iter = {0:d} \nrxyz_final = \n{1:s}".\
                  format(i_iter+1, np.array_str(rxyz_new, precision=6, suppress_small=False)) )
            print ( "Forces = \n{0:s}".\
                  format(np.array_str(del_fp, precision=6, suppress_small=False)) )
            print ( "Finger print energy difference = {0:s}".\
                  format(np.array_str(fp_dist, precision=6, suppress_small=False)) )
            '''
    
    return del_fp

# Calculate forces using finite difference method
def get_FD_forces(lat, rxyz, types, contract = False, ntyp = 1, nx = 300, \
                  lmax = 0, znucl = np.array([3], int), cutoff = 6.5, \
                  iter_max = 1, step_size = 1e-4):
    '''
    ntyp = 1
    nx = 300
    lmax = 0
    cutoff = 6.5
    znucl = np.array([3], int)
    lat, rxyz, types = readvasp(v1)
    contract = False
    i_iter = 0
    iter_max = 4
    atol = 1.0e-6
    step_size = 1e-4
    const_factor = 1.0e+31
    '''
    del_fp_dist = 0.0
    rxyz_left = rxyz.copy()
    rxyz_new = rxyz.copy()
    rxyz_right = rxyz.copy()
    rxyz_delta = np.zeros_like(rxyz)
    for i_iter in range(iter_max):
        del_fp = np.zeros((len(rxyz_new), 3))
        finite_diff = np.zeros((len(rxyz_new), 3))
        sum_del_fp = np.zeros(3)
        fp_dist_0 = 0.0
        fp_dist_new = 0.0
        fp_dist_del = 0.0
        rxyz_delta = step_size*get_rxyz_delta(rxyz)
        rxyz_new = np.add(rxyz_new, rxyz_delta)
        for i_atom in range(len(rxyz)):
            for j_atom in range(len(rxyz)):
                for k in range(3):
                    h = rxyz_delta[i_atom][k]
                    # rxyz_left[i_atom][k] = rxyz_left[i_atom][k] - 2.0*h
                    rxyz_right[i_atom][k] = rxyz_right[i_atom][k] + 2.0*h
                    
                    fp_iat_left = \
                    get_fp(contract, ntyp, nx, lmax, lat, \
                                              rxyz_left, types, znucl, cutoff, i_atom)
                    fp_jat_left = \
                    get_fp(contract, ntyp, nx, lmax, lat, \
                                              rxyz_left, types, znucl, cutoff, j_atom)
                    fp_iat = \
                    get_fp(contract, ntyp, nx, lmax, lat, \
                                              rxyz_new, types, znucl, cutoff, i_atom)
                    fp_jat = \
                    get_fp(contract, ntyp, nx, lmax, lat, \
                                              rxyz_new, types, znucl, cutoff, j_atom)
                    fp_iat_right = \
                    get_fp(contract, ntyp, nx, lmax, lat, \
                                              rxyz_right, types, znucl, cutoff, i_atom)
                    fp_jat_right = \
                    get_fp(contract, ntyp, nx, lmax, lat, \
                                              rxyz_right, types, znucl, cutoff, j_atom)
                    
                    fp_dist_left = get_fpdist(ntyp, types, fp_iat_left, fp_jat_left)
                    fp_dist_right = get_fpdist(ntyp, types, fp_iat_right, fp_jat_right)
                    finite_diff[i_atom][k] = (fp_dist_right - fp_dist_left)/(2.0*h)
                    
        
        # sum_del_fp = np.sum(del_fp, axis=0)
        sum_finite_diff = np.sum(finite_diff, axis=0)
        for ii_atom in range(len(rxyz_new)):
            # del_fp[ii_atom, :] = del_fp[ii_atom, :] - sum_del_fp/len(rxyz_new)
            finite_diff[ii_atom, :] = finite_diff[ii_atom, :] - sum_finite_diff/len(rxyz)
        
        '''
        print ( "i_iter = {0:d} \nrxyz_new = \n{1:s}".\
              format(i_iter+1, np.array_str(rxyz_new, precision=6, suppress_small=False)) )
        print ( "Forces = \n{0:s}".\
              format(np.array_str(del_fp, precision=6, suppress_small=False)) )
        print ( "Finite difference = \n{0:s}".\
              format(np.array_str(finite_diff, precision=6, suppress_small=False)) )
        '''
    
    return finite_diff




# Calculate numerical inegration using Simpson's rule
def get_simpson_energy(lat, rxyz, types, contract = False, ntyp = 1, nx = 300, \
                       lmax = 0, znucl = np.array([3], int), cutoff = 6.5, \
                       iter_max = 1, step_size = 1e-4):
    '''
    ntyp = 1
    nx = 300
    lmax = 0
    cutoff = 6.5
    znucl = np.array([3], int)
    lat, rxyz, types = readvasp(v1)
    contract = False
    i_iter = 0
    iter_max = 4
    atol = 1.0e-6
    step_size = 1e-4
    const_factor = 1.0e+31
    '''
    del_fp_dist = 0.0
    rxyz_left = rxyz.copy()
    rxyz_new = rxyz.copy()
    rxyz_right = rxyz.copy()
    rxyz_delta = np.zeros_like(rxyz)
    for i_iter in range(iter_max):
        del_fp_left = np.zeros((len(rxyz_new), 3))
        del_fp = np.zeros((len(rxyz_new), 3))
        del_fp_right = np.zeros((len(rxyz_new), 3))
        sum_del_fp_left = np.zeros(3)
        sum_del_fp = np.zeros(3)
        sum_del_fp_right = np.zeros(3)
        fp_dist_0 = 0.0
        fp_dist_new = 0.0
        fp_dist_del = 0.0
        rxyz_delta = step_size*get_rxyz_delta(rxyz)
        rxyz_left = rxyz_new.copy()
        rxyz_new = np.add(rxyz_new, rxyz_delta)
        rxyz_right = np.add(rxyz_new, rxyz_delta)
        for i_atom in range(len(rxyz)):
            # del_fp = np.zeros(3)
            for j_atom in range(len(rxyz)):
                fp_iat_0 = \
                get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff, i_atom)
                fp_jat_0 = \
                get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff, j_atom)
                fp_iat_left = \
                get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_left, types, znucl, cutoff, i_atom)
                fp_jat_left = \
                get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_left, types, znucl, cutoff, j_atom)
                fp_iat = \
                get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, i_atom)
                fp_jat = \
                get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, j_atom)
                fp_iat_right = \
                get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_right, types, znucl, cutoff, i_atom)
                fp_jat_right = \
                get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_right, types, znucl, cutoff, j_atom)
                D_fp_mat_iat_left = \
                get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_left, types, znucl, cutoff, i_atom)
                D_fp_mat_jat_left = \
                get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_left, types, znucl, cutoff, j_atom)
                D_fp_mat_iat = \
                get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, i_atom)
                D_fp_mat_jat = \
                get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, j_atom)
                D_fp_mat_iat_right = \
                get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_right, types, znucl, cutoff, i_atom)
                D_fp_mat_jat_right = \
                get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_right, types, znucl, cutoff, j_atom)
                diff_fp_left = fp_iat_left - fp_jat_left
                diff_fp = fp_iat - fp_jat
                diff_fp_right = fp_iat_right - fp_jat_right
                # common_count, i_rxyz_sphere_1, i_rxyz_sphere_2 = \
                # fplib_GD.get_common_sphere(ntyp, nx, lmax, lat, rxyz, types, \
                #                                 znucl, cutoff, i_atom, j_atom)
                iat_in_j_sphere_left, iat_j_left = get_common_sphere(ntyp, \
                              nx, lmax, lat, rxyz_left, types, znucl, cutoff, i_atom, j_atom)
                if iat_in_j_sphere_left:
                    diff_D_fp_x_left = D_fp_mat_iat_left[i_atom, 0, :] - \
                                       D_fp_mat_jat_left[iat_j_left, 0, :]
                    diff_D_fp_y_left = D_fp_mat_iat_left[i_atom, 1, :] - \
                                       D_fp_mat_jat_left[iat_j_left, 1, :]
                    diff_D_fp_z_left = D_fp_mat_iat_left[i_atom, 2, :] - \
                                       D_fp_mat_jat_left[iat_j_left, 2, :]
                else:
                    diff_D_fp_x_left = D_fp_mat_iat_left[i_atom, 0, :]
                    diff_D_fp_y_left = D_fp_mat_iat_left[i_atom, 1, :]
                    diff_D_fp_z_left = D_fp_mat_iat_left[i_atom, 2, :]
                
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
                
                iat_in_j_sphere_right, iat_j_right = get_common_sphere(ntyp, \
                              nx, lmax, lat, rxyz_right, types, znucl, cutoff, i_atom, j_atom)
                if iat_in_j_sphere_right:
                    diff_D_fp_x_right = D_fp_mat_iat_right[i_atom, 0, :] - \
                                        D_fp_mat_jat_right[iat_j_right, 0, :]
                    diff_D_fp_y_right = D_fp_mat_iat_right[i_atom, 1, :] - \
                                        D_fp_mat_jat_right[iat_j_right, 1, :]
                    diff_D_fp_z_right = D_fp_mat_iat_right[i_atom, 2, :] - \
                                        D_fp_mat_jat_right[iat_j_right, 2, :]
                else:
                    diff_D_fp_x_right = D_fp_mat_iat_right[i_atom, 0, :]
                    diff_D_fp_y_right = D_fp_mat_iat_right[i_atom, 1, :]
                    diff_D_fp_z_right = D_fp_mat_iat_right[i_atom, 2, :]
                
                diff_D_fp_x_left = np.vstack( (np.array(diff_D_fp_x_left)[::-1], ) ).T
                diff_D_fp_y_left = np.vstack( (np.array(diff_D_fp_y_left)[::-1], ) ).T
                diff_D_fp_z_left = np.vstack( (np.array(diff_D_fp_z_left)[::-1], ) ).T
                diff_D_fp_x = np.vstack( (np.array(diff_D_fp_x)[::-1], ) ).T
                diff_D_fp_y = np.vstack( (np.array(diff_D_fp_y)[::-1], ) ).T
                diff_D_fp_z = np.vstack( (np.array(diff_D_fp_z)[::-1], ) ).T
                diff_D_fp_x_right = np.vstack( (np.array(diff_D_fp_x_right)[::-1], ) ).T
                diff_D_fp_y_right = np.vstack( (np.array(diff_D_fp_y_right)[::-1], ) ).T
                diff_D_fp_z_right = np.vstack( (np.array(diff_D_fp_z_right)[::-1], ) ).T
                
                del_fp_left[i_atom][0] = del_fp_left[i_atom][0] + \
                                    2.0*np.real( np.matmul( diff_fp_left.T, diff_D_fp_x_left ) )
                del_fp_left[i_atom][1] = del_fp_left[i_atom][1] + \
                                    2.0*np.real( np.matmul( diff_fp_left.T, diff_D_fp_y_left ) )
                del_fp_left[i_atom][2] = del_fp_left[i_atom][2] + \
                                    2.0*np.real( np.matmul( diff_fp_left.T, diff_D_fp_z_left ) )
                del_fp[i_atom][0] = del_fp[i_atom][0] + \
                                    2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_x ) )
                del_fp[i_atom][1] = del_fp[i_atom][1] + \
                                    2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_y ) )
                del_fp[i_atom][2] = del_fp[i_atom][2] + \
                                    2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_z ) )
                del_fp_right[i_atom][0] = del_fp_right[i_atom][0] + \
                                    2.0*np.real( np.matmul( diff_fp_right.T, diff_D_fp_x_right ) )
                del_fp_right[i_atom][1] = del_fp_right[i_atom][1] + \
                                    2.0*np.real( np.matmul( diff_fp_right.T, diff_D_fp_y_right ) )
                del_fp_right[i_atom][2] = del_fp_right[i_atom][2] + \
                                    2.0*np.real( np.matmul( diff_fp_right.T, diff_D_fp_z_right ) )
                
                fp_dist_0 = fp_dist_0 + get_fpdist(ntyp, types, fp_iat_0, fp_jat_0)
                fp_dist_new = fp_dist_new + get_fpdist(ntyp, types, fp_iat, fp_jat)
                fp_dist_del = fp_dist_new - fp_dist_0
                del_fp_dist = del_fp_dist + np.vdot(rxyz_delta[i_atom], del_fp[i_atom])
                
                
            # del_fp_dist = del_fp_dist + np.vdot(rxyz_delta[i_atom], del_fp[i_atom])
        
        sum_del_fp_left = np.sum(del_fp_left, axis=0)
        sum_del_fp = np.sum(del_fp, axis=0)
        sum_del_fp_right = np.sum(del_fp_right, axis=0)
        
        for ii_atom in range(len(rxyz_new)):
            del_fp_left[ii_atom, :] = del_fp_left[ii_atom, :] - sum_del_fp_left/len(rxyz_new)
            del_fp[ii_atom, :] = del_fp[ii_atom, :] - sum_del_fp/len(rxyz_new)
            del_fp_right[ii_atom, :] = del_fp_right[ii_atom, :] - sum_del_fp_right/len(rxyz_new)
            del_fp_dist = del_fp_dist + \
                          ( np.absolute( np.dot(rxyz_delta[ii_atom], del_fp_left[ii_atom]) ) + \
                        4.0*np.absolute( np.dot(rxyz_delta[ii_atom], del_fp[ii_atom]) ) + \
                            np.absolute( np.dot(rxyz_delta[ii_atom], del_fp_right[ii_atom]) ) )/3.0
        
        '''
        print ( "i_iter = {0:d} \nrxyz_new = \n{1:s}".\
              format(i_iter+1, np.array_str(rxyz_new, precision=6, suppress_small=False)) )
        print ( "Numerical integral = {0:.6e}".format(del_fp_dist) )
        print ( "Forces = \n{0:s}".\
              format(np.array_str(del_fp, precision=6, suppress_small=False)) )
        print ( "Finger print energy difference = {0:s}".\
              format(np.array_str(fp_dist_del, precision=6, suppress_small=False)) )
        '''
    
    
    return del_fp_dist

# Calculate Cauchy stress tensor using finite difference
def get_FD_stress(lat, pos, types, contract = False, ntyp = 1, nx = 300, \
                  lmax = 0, znucl = np.array([3], int), cutoff = 6.5, \
                  iter_max = 1, step_size = 1e-4):
    '''
    ntyp = 1
    nx = 300
    lmax = 0
    cutoff = 6.5
    znucl = np.array([3], int)
    lat, rxyz, types = readvasp(v1)
    contract = False
    i_iter = 0
    iter_max = 4
    atol = 1.0e-6
    step_size = 1e-4
    const_factor = 1.0e+31
    '''
    rxyz = np.dot(pos, lat)
    fp_energy = 0.0
    fp_energy_new = 0.0
    fp_energy_left = 0.0
    fp_energy_right = 0.0
    # cell_vol = 0.0
    # lat_new = lat.copy()
    # lat_left = lat.copy()
    # lat_right = lat.copy()
    rxyz = np.dot(pos, lat)
    # rxyz_new = np.dot(pos, lat_new)
    # rxyz_left = np.dot(pos, lat_left)
    # rxyz_right = np.dot(pos, lat_right)
    rxyz_delta = np.zeros_like(rxyz)
    for i_iter in range(iter_max):
        cell_vol = np.inner( lat[0], np.cross( lat[1], lat[2] ) )
        stress = np.zeros((3, 3))
        fp_energy = get_fp_energy(lat, rxyz, types, contract = False, ntyp = 1, nx = 300, \
                                        lmax = 0, znucl = np.array([3], int), cutoff = 6.5)
        strain_delta_tmp = step_size*np.random.randint(1, 9999, (3, 3))/9999
        # Make strain tensor symmetric
        strain_delta = 0.5*(strain_delta_tmp + strain_delta_tmp.T - \
                            np.diag(np.diag(strain_delta_tmp))) 
        rxyz_ratio = np.diag(np.ones(3))
        rxyz_ratio_new = rxyz_ratio.copy()
        for m in range(3):
            for n in range(3):
                h = strain_delta[m][n]
                rxyz_ratio_left = np.diag(np.ones(3))
                rxyz_ratio_right = np.diag(np.ones(3))
                rxyz_ratio_left[m][n] = rxyz_ratio[m][n] - h
                rxyz_ratio_right[m][n] = rxyz_ratio[m][n] + h
                lat_left = np.multiply(lat, rxyz_ratio_left.T)
                lat_right = np.multiply(lat, rxyz_ratio_right.T)
                rxyz_left = np.dot(pos, lat_left)
                rxyz_right = np.dot(pos, lat_right)
                fp_energy_left = get_fp_energy(lat_left, rxyz_left, types, contract = False, \
                                               ntyp = 1, nx = 300, lmax = 0, znucl = \
                                               np.array([3], int), cutoff = 6.5)
                fp_energy_right = get_fp_energy(lat_right, rxyz_right, types, contract = False, \
                                                ntyp = 1, nx = 300, lmax = 0, znucl = \
                                                np.array([3], int), cutoff = 6.5)
                stress[m][n] = - (fp_energy_right - fp_energy_left)/(2.0*h*cell_vol)
        #################
        
    #################
    stress = stress.flat[[0, 4, 8, 5, 2, 1]]
    return stress