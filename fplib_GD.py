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
                # if D_n == iat:
                #     Dx_om[iat][jat] = -(2.0*r) * d[0] * sji * amp[iat] * amp[jat]
                # elif D_n == jat:
                #     Dx_om[iat][jat] =  (2.0*r) * d[0] * sji * amp[iat] * amp[jat]
                # else:
                #     Dx_om[iat][jat] = 0.0
                
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
                # if D_n == 4*iat and D_n != 4*jat and D_n != 4*jat+1 and  \
                # D_n != 4*jat+2 and D_n != 4*jat:
                #     Dx_om[4*iat][4*jat] = -(2.0*r) * d[0] * sji * amp[iat] * amp[jat]
                # elif D_n == 4*jat and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dx_om[4*iat][4*jat] =  (2.0*r) * d[0] * sji * amp[iat] * amp[jat]
                # else:
                #     Dx_om[4*iat][4*jat] = 0.0
                
                # Derivative of <s_i | p_j>
                stv = np.sqrt(8.0) * rcov[jat] * r * sji
                Dx_om[4*iat][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[0], d[0] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dx_om[4*iat][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[1], d[1] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dx_om[4*iat][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[2], d[2] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                # if D_n == 4*jat+1 and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dx_om[4*iat][4*jat+1] = stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[0], d[0] ) * 2.0*r * amp[iat] * amp[jat] 
                # elif D_n == 4*jat+2 and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dx_om[4*iat][4*jat+2] = stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[0], d[0] ) * 2.0*r * amp[iat] * amp[jat] 
                # elif D_n == 4*jat+3 and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dx_om[4*iat][4*jat+3] = stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[0], d[0] ) * 2.0*r * amp[iat] * amp[jat] 
                # elif D_n == 4*iat and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dx_om[4*iat][4*jat+1] = - stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[0], d[0] ) * 2.0*r * amp[iat] * amp[jat] 
                #     Dx_om[4*iat][4*jat+2] = - stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[1], d[1] ) * 2.0*r * amp[iat] * amp[jat] 
                #     Dx_om[4*iat][4*jat+3] = - stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[2], d[2] ) * 2.0*r * amp[iat] * amp[jat] 
                # else:
                #     Dx_om[4*iat][4*jat+1] = 0.0
                #     Dx_om[4*iat][4*jat+2] = 0.0
                #     Dx_om[4*iat][4*jat+3] = 0.0

                # Derivative of <p_i | s_j>
                stv = np.sqrt(8.0) * rcov[iat] * r * sji * -1.0
                Dx_om[4*iat+1][4*jat] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[0], d[0] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dx_om[4*iat+2][4*jat] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[1], d[1] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dx_om[4*iat+3][4*jat] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[2], d[2] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                # if D_n == 4*iat+1 and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dx_om[4*iat+1][4*jat] = stv * amp[iat] * amp[jat] \
                #     - stv * np.dot( d[0], d[0] ) * 2.0*r * amp[iat] * amp[jat] 
                # elif D_n == 4*iat+2 and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dx_om[4*iat+2][4*jat] = stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[0], d[0] ) * 2.0*r * amp[iat] * amp[jat] 
                # elif D_n == 4*iat+3 and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dx_om[4*iat+3][4*jat] = stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[0], d[0] ) * 2.0*r * amp[iat] * amp[jat] 
                # elif D_n == 4*jat and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dx_om[4*iat+1][4*jat] = - stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[0], d[0] ) * 2.0*r * amp[iat] * amp[jat] 
                #     Dx_om[4*iat+2][4*jat] = - stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[1], d[1] ) * 2.0*r * amp[iat] * amp[jat] 
                #     Dx_om[4*iat+3][4*jat] = - stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[2], d[2] ) * 2.0*r * amp[iat] * amp[jat] 
                # else:
                #     Dx_om[4*iat+1][4*jat] = 0.0
                #     Dx_om[4*iat+2][4*jat] = 0.0
                #     Dx_om[4*iat+3][4*jat] = 0.0

                # Derivative of <p_i | p_j>
                stv = -8.0 * rcov[iat] * rcov[jat] * r * r * sji
                Dx_om[4*iat+1][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[0] * d[0] - 0.5/r) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[0] + d[0]) \
                                                                     * amp[iat] * amp[jat]
                Dx_om[4*iat+1][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[1] * d[0]        ) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[1] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dx_om[4*iat+1][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[2] * d[0]        ) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[2] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dx_om[4*iat+2][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[0] * d[1]        ) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[0] + d[0]) \
                                                                     * amp[iat] * amp[jat]
                Dx_om[4*iat+2][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[1] * d[1] - 0.5/r) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[1] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dx_om[4*iat+2][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[2] * d[1]        ) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[2] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dx_om[4*iat+3][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[0] * d[2]        ) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[0] + d[0]) \
                                                                     * amp[iat] * amp[jat]
                Dx_om[4*iat+3][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[1] * d[2]        ) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[1] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dx_om[4*iat+3][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[2] * d[2] - 0.5/r) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[2] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                # if D_n == 4*iat+1 and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dx_om[4*iat+1][4*jat+1] = -(2.0*r) * d[0] * stv \
                #     * (d[0] * d[0] - 0.5/r) * amp[iat] * amp[jat] + \
                #     stv * (2.0 * d[0]        ) * amp[iat] * amp[jat]
                #     Dx_om[4*iat+1][4*jat+2] = -(2.0*r) * d[0] * stv \
                #     * (d[1] * d[0]        ) * amp[iat] * amp[jat] + \
                #     stv * (1.0 * d[1]        ) * amp[iat] * amp[jat]
                #     Dx_om[4*iat+1][4*jat+3] = -(2.0*r) * d[0] * stv \
                #     * (d[2] * d[0]        ) * amp[iat] * amp[jat] + \
                #     stv * (1.0 * d[2]        ) * amp[iat] * amp[jat]
                # elif D_n == 4*iat+2 and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dx_om[4*iat+2][4*jat+1] = -(2.0*r) * d[0] * stv \
                #     * (d[0] * d[1]        ) * amp[iat] * amp[jat] + \
                #     stv * (1.0 * d[1]        ) * amp[iat] * amp[jat]
                #     Dx_om[4*iat+2][4*jat+2] = -(2.0*r) * d[0] * stv \
                #     * (d[1] * d[1] - 0.5/r) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0        ) * amp[iat] * amp[jat]
                #     Dx_om[4*iat+2][4*jat+3] = -(2.0*r) * d[0] * stv \
                #     * (d[2] * d[1]        ) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0        ) * amp[iat] * amp[jat]
                # elif D_n == 4*iat+3 and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dx_om[4*iat+3][4*jat+1] = -(2.0*r) * d[0] * stv \
                #     * (d[0] * d[2]        ) * amp[iat] * amp[jat] + \
                #     stv * (1.0 * d[2]        ) * amp[iat] * amp[jat]
                #     Dx_om[4*iat+3][4*jat+2] = -(2.0*r) * d[0] * stv \
                #     * (d[1] * d[2]        ) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0        ) * amp[iat] * amp[jat]
                #     Dx_om[4*iat+3][4*jat+3] = -(2.0*r) * d[0] * stv \
                #     * (d[2] * d[2] - 0.5/r) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0        ) * amp[iat] * amp[jat]
                # elif D_n == 4*jat+1 and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dx_om[4*iat+1][4*jat+1] = (2.0*r) * d[0] * stv \
                #     * (d[0] * d[0] - 0.5/r) * amp[iat] * amp[jat] + \
                #     stv * (-2.0 * d[0]        ) * amp[iat] * amp[jat]
                #     Dx_om[4*iat+2][4*jat+1] = (2.0*r) * d[0] * stv \
                #     * (d[1] * d[0]        ) * amp[iat] * amp[jat] + \
                #     stv * (-1.0 * d[1]        ) * amp[iat] * amp[jat]
                #     Dx_om[4*iat+3][4*jat+1] = (2.0*r) * d[0] * stv \
                #     * (d[2] * d[0]        ) * amp[iat] * amp[jat] + \
                #     stv * (-1.0 * d[2]        ) * amp[iat] * amp[jat]
                # elif D_n == 4*jat+2 and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dx_om[4*iat+1][4*jat+2] = (2.0*r) * d[0] * stv \
                #     * (d[0] * d[1]        ) * amp[iat] * amp[jat] + \
                #     stv * (-1.0 * d[1]        ) * amp[iat] * amp[jat]
                #     Dx_om[4*iat+2][4*jat+2] = (2.0*r) * d[0] * stv \
                #     * (d[1] * d[1] - 0.5/r) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0        ) * amp[iat] * amp[jat]
                #     Dx_om[4*iat+3][4*jat+2] = (2.0*r) * d[0] * stv \
                #     * (d[2] * d[1]        ) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0        ) * amp[iat] * amp[jat]
                # elif D_n == 4*jat+3 and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dx_om[4*iat+1][4*jat+3] = (2.0*r) * d[0] * stv \
                #     * (d[0] * d[2]        ) * amp[iat] * amp[jat] + \
                #     stv * (-1.0 * d[2]        ) * amp[iat] * amp[jat]
                #     Dx_om[4*iat+2][4*jat+3] = (2.0*r) * d[0] * stv \
                #     * (d[1] * d[2]        ) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0        ) * amp[iat] * amp[jat]
                #     Dx_om[4*iat+3][4*jat+3] = (2.0*r) * d[0] * stv \
                #     * (d[2] * d[2] - 0.5/r) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0        ) * amp[iat] * amp[jat]
                # else:
                #     Dx_om[4*iat+1][4*jat+1] = 0.0
                #     Dx_om[4*iat+1][4*jat+2] = 0.0
                #     Dx_om[4*iat+1][4*jat+3] = 0.0
                #     Dx_om[4*iat+2][4*jat+1] = 0.0
                #     Dx_om[4*iat+2][4*jat+2] = 0.0
                #     Dx_om[4*iat+2][4*jat+3] = 0.0
                #     Dx_om[4*iat+3][4*jat+1] = 0.0
                #     Dx_om[4*iat+3][4*jat+2] = 0.0
                #     Dx_om[4*iat+3][4*jat+3] = 0.0
                
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
                # if D_n == iat:
                #     Dy_om[iat][jat] = -(2.0*r) * d[1] * sji * amp[iat] * amp[jat]
                # elif D_n == jat:
                #     Dy_om[iat][jat] =  (2.0*r) * d[1] * sji * amp[iat] * amp[jat]
                # else:
                #     Dy_om[iat][jat] = 0.0
                
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
                # if D_n == 4*iat and D_n != 4*jat and D_n != 4*jat+1 and  \
                # D_n != 4*jat+2 and D_n != 4*jat:
                #     Dy_om[4*iat][4*jat] = -(2.0*r) * d[1] * sji * amp[iat] * amp[jat]
                # elif D_n == 4*jat and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dy_om[4*iat][4*jat] =  (2.0*r) * d[1] * sji * amp[iat] * amp[jat]
                # else:
                #     Dy_om[4*iat][4*jat] = 0.0
                
                # Derivative of <s_i | p_j>
                stv = np.sqrt(8.0) * rcov[jat] * r * sji
                Dy_om[4*iat][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[0], d[0] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dy_om[4*iat][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[1], d[1] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dy_om[4*iat][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[2], d[2] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                # if D_n == 4*jat+1 and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dy_om[4*iat][4*jat+1] = stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[1], d[1] ) * 2.0*r * amp[iat] * amp[jat] 
                # elif D_n == 4*jat+2 and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dy_om[4*iat][4*jat+2] = stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[1], d[1] ) * 2.0*r * amp[iat] * amp[jat] 
                # elif D_n == 4*jat+3 and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dy_om[4*iat][4*jat+3] = stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[1], d[1] ) * 2.0*r * amp[iat] * amp[jat] 
                # elif D_n == 4*iat and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dy_om[4*iat][4*jat+1] = - stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[0], d[0] ) * 2.0*r * amp[iat] * amp[jat] 
                #     Dy_om[4*iat][4*jat+2] = - stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[1], d[1] ) * 2.0*r * amp[iat] * amp[jat] 
                #    Dy_om[4*iat][4*jat+3] = - stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[2], d[2] ) * 2.0*r * amp[iat] * amp[jat] 
                # else:
                #     Dy_om[4*iat][4*jat+1] = 0.0
                #     Dy_om[4*iat][4*jat+2] = 0.0
                #     Dy_om[4*iat][4*jat+3] = 0.0

                # Derivative of <p_i | s_j>
                stv = np.sqrt(8.0) * rcov[iat] * r * sji * -1.0
                Dy_om[4*iat+1][4*jat] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[0], d[0] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dy_om[4*iat+2][4*jat] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[1], d[1] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dy_om[4*iat+3][4*jat] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[2], d[2] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                # if D_n == 4*iat+1 and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dy_om[4*iat+1][4*jat] = stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[1], d[1] ) * 2.0*r * amp[iat] * amp[jat] 
                # elif D_n == 4*iat+2 and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dy_om[4*iat+2][4*jat] = stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[1], d[1] ) * 2.0*r * amp[iat] * amp[jat] 
                # elif D_n == 4*iat+3 and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dy_om[4*iat+3][4*jat] = stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[1], d[1] ) * 2.0*r * amp[iat] * amp[jat] 
                # elif D_n == 4*jat and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dy_om[4*iat+1][4*jat] = - stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[0], d[0] ) * 2.0*r * amp[iat] * amp[jat] 
                #     Dy_om[4*iat+2][4*jat] = - stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[1], d[1] ) * 2.0*r * amp[iat] * amp[jat] 
                #     Dy_om[4*iat+3][4*jat] = - stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[2], d[2] ) * 2.0*r * amp[iat] * amp[jat] 
                # else:
                #     Dy_om[4*iat+1][4*jat] = 0.0
                #     Dy_om[4*iat+2][4*jat] = 0.0
                #     Dy_om[4*iat+3][4*jat] = 0.0

                # Derivative of <p_i | p_j>
                stv = -8.0 * rcov[iat] * rcov[jat] * r * r * sji
                Dy_om[4*iat+1][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[0] * d[0] - 0.5/r) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[0] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dy_om[4*iat+1][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[1] * d[0]        ) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[1] + d[1]) \
                                                                     * amp[iat] * amp[jat]
                Dy_om[4*iat+1][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[2] * d[0]        ) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[2] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dy_om[4*iat+2][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[0] * d[1]        ) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[0] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dy_om[4*iat+2][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[1] * d[1] - 0.5/r) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[1] + d[1]) \
                                                                     * amp[iat] * amp[jat]
                Dy_om[4*iat+2][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[2] * d[1]        ) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[2] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dy_om[4*iat+3][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[0] * d[2]        ) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[0] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dy_om[4*iat+3][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[1] * d[2]        ) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[1] + d[1]) \
                                                                     * amp[iat] * amp[jat]
                Dy_om[4*iat+3][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[2] * d[2] - 0.5/r) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[2] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                # if D_n == 4*iat+1 and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dy_om[4*iat+1][4*jat+1] = -(4.0*r) * d[1] * stv \
                #     * (d[0] * d[0] - 0.5/r) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                #     Dx_om[4*iat+1][4*jat+2] = -(4.0*r) * d[1] * stv \
                #     * (d[1] * d[0]        ) * amp[iat] * amp[jat] + \
                #     stv * (1.0 * d[0]        ) * amp[iat] * amp[jat]
                #     Dx_om[4*iat+1][4*jat+3] = -(4.0*r) * d[1] * stv \
                #     * (d[2] * d[0]        ) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                # elif D_n == 4*iat+2 and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dy_om[4*iat+2][4*jat+1] = -(4.0*r) * d[1] * stv \
                #     * (d[0] * d[1]        ) * amp[iat] * amp[jat] + \
                #     stv * (1.0 * d[0]        ) * amp[iat] * amp[jat]
                #     Dy_om[4*iat+2][4*jat+2] = -(4.0*r) * d[1] * stv \
                #     * (d[1] * d[1] - 0.5/r) * amp[iat] * amp[jat] + \
                #     stv * (2.0 * d[1]        ) * amp[iat] * amp[jat]
                #     Dy_om[4*iat+2][4*jat+3] = -(4.0*r) * d[1] * stv \
                #     * (d[2] * d[1]        ) * amp[iat] * amp[jat] + \
                #     stv * (1.0 * d[2]        ) * amp[iat] * amp[jat]
                # elif D_n == 4*iat+3 and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dy_om[4*iat+3][4*jat+1] = -(4.0*r) * d[1] * stv \
                #     * (d[0] * d[2]        ) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                #     Dy_om[4*iat+3][4*jat+2] = -(4.0*r) * d[1] * stv \
                #     * (d[1] * d[2]        ) * amp[iat] * amp[jat] + \
                #     stv * (1.0 * d[2]       ) * amp[iat] * amp[jat]
                #     Dy_om[4*iat+3][4*jat+3] = -(4.0*r) * d[1] * stv \
                #     * (d[2] * d[2] - 0.5/r) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                # elif D_n == 4*jat+1 and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dy_om[4*iat+1][4*jat+1] = (4.0*r) * d[1] * stv \
                #     * (d[0] * d[0] - 0.5/r) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0          ) * amp[iat] * amp[jat]
                #     Dy_om[4*iat+2][4*jat+1] = (4.0*r) * d[1] * stv \
                #     * (d[1] * d[0]        ) * amp[iat] * amp[jat] + \
                #     stv * (-1.0 * d[0]        ) * amp[iat] * amp[jat]
                #     Dy_om[4*iat+3][4*jat+1] = (4.0*r) * d[1] * stv \
                #     * (d[2] * d[0]        ) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0          ) * amp[iat] * amp[jat]
                # elif D_n == 4*jat+2 and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dy_om[4*iat+1][4*jat+2] = (4.0*r) * d[1] * stv \
                #     * (d[0] * d[1]        ) * amp[iat] * amp[jat] + \
                #     stv * (-1.0 * d[0]        ) * amp[iat] * amp[jat]
                #     Dy_om[4*iat+2][4*jat+2] = (4.0*r) * d[1] * stv \
                #     * (d[1] * d[1] - 0.5/r) * amp[iat] * amp[jat] + \
                #     stv * (-2.0 * d[1]       ) * amp[iat] * amp[jat]
                #     Dy_om[4*iat+3][4*jat+2] = (4.0*r) * d[1] * stv \
                #     * (d[2] * d[1]        ) * amp[iat] * amp[jat] + \
                #     stv * (-1.0 * d[2]        ) * amp[iat] * amp[jat]
                # elif D_n == 4*jat+3 and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dy_om[4*iat+1][4*jat+3] = (4.0*r) * d[1] * stv \
                #     * (d[0] * d[2]        ) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0          ) * amp[iat] * amp[jat]
                #     Dy_om[4*iat+2][4*jat+3] = (4.0*r) * d[1] * stv \
                #     * (d[1] * d[2]        ) * amp[iat] * amp[jat] + \
                #     stv * (-1.0 * d[2]        ) * amp[iat] * amp[jat]
                #     Dy_om[4*iat+3][4*jat+3] = (4.0*r) * d[1] * stv \
                #     * (d[2] * d[2] - 0.5/r) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                # else:
                #     Dy_om[4*iat+1][4*jat+1] = 0.0
                #     Dy_om[4*iat+1][4*jat+2] = 0.0
                #     Dy_om[4*iat+1][4*jat+3] = 0.0
                #     Dy_om[4*iat+2][4*jat+1] = 0.0
                #     Dy_om[4*iat+2][4*jat+2] = 0.0
                #     Dy_om[4*iat+2][4*jat+3] = 0.0
                #     Dy_om[4*iat+3][4*jat+1] = 0.0
                #     Dy_om[4*iat+3][4*jat+2] = 0.0
                #     Dy_om[4*iat+3][4*jat+3] = 0.0
                
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
                # if D_n == iat:
                #     Dz_om[iat][jat] = -(2.0*r) * d[2] * sji * amp[iat] * amp[jat]
                # elif D_n == jat:
                #     Dz_om[iat][jat] =  (2.0*r) * d[2] * sji * amp[iat] * amp[jat]
                # else:
                #     Dz_om[iat][jat] = 0.0
                
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
                # if D_n == 4*iat and D_n != 4*jat and D_n != 4*jat+1 and  \
                # D_n != 4*jat+2 and D_n != 4*jat:
                #     Dz_om[4*iat][4*jat] = -(4.0*r) * d[2] * sji * amp[iat] * amp[jat]
                # elif D_n == 4*jat and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dz_om[4*iat][4*jat] =  (4.0*r) * d[2] * sji * amp[iat] * amp[jat]
                # else:
                #     Dz_om[4*iat][4*jat] = 0.0
                
                # Derivative of <s_i | p_j>
                stv = np.sqrt(8.0) * rcov[jat] * r * sji
                Dz_om[4*iat][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[0], d[0] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dz_om[4*iat][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[1], d[1] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dz_om[4*iat][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[2], d[2] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                # if D_n == 4*jat+1 and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dz_om[4*iat][4*jat+1] = stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                # elif D_n == 4*jat+2 and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dz_om[4*iat][4*jat+2] = stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                # elif D_n == 4*jat+3 and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dz_om[4*iat][4*jat+3] = stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                # elif D_n == 4*iat and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dz_om[4*iat][4*jat+1] = - stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[0], d[0] ) * 4.0*r * amp[iat] * amp[jat] 
                #     Dz_om[4*iat][4*jat+2] = - stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[1], d[1] ) * 4.0*r * amp[iat] * amp[jat] 
                #     Dz_om[4*iat][4*jat+3] = - stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                # else:
                #     Dz_om[4*iat][4*jat+1] = 0.0
                #     Dz_om[4*iat][4*jat+2] = 0.0
                #     Dz_om[4*iat][4*jat+3] = 0.0

                # Derivative of <p_i | s_j>
                stv = np.sqrt(8.0) * rcov[iat] * r * sji * -1.0
                Dz_om[4*iat+1][4*jat] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[0], d[0] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dz_om[4*iat+2][4*jat] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[1], d[1] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                Dz_om[4*iat+3][4*jat] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                * ( stv * amp[iat] * amp[jat] - stv * np.dot( d[2], d[2] ) * 2.0*r \
                                                                 * amp[iat] * amp[jat] )
                # if D_n == 4*iat+1 and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dz_om[4*iat+1][4*jat] = stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                # elif D_n == 4*iat+2 and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dz_om[4*iat+2][4*jat] = stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                # elif D_n == 4*iat+3 and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dz_om[4*iat+3][4*jat] = stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                # elif D_n == 4*jat and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dz_om[4*iat+1][4*jat] = - stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[0], d[0] ) * 4.0*r * amp[iat] * amp[jat] 
                #     Dz_om[4*iat+2][4*jat] = - stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[1], d[1] ) * 4.0*r * amp[iat] * amp[jat] 
                #     Dz_om[4*iat+3][4*jat] = - stv * amp[iat] * amp[jat] \
                #     + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                # else:
                #     Dz_om[4*iat+1][4*jat] = 0.0
                #     Dz_om[4*iat+2][4*jat] = 0.0
                #     Dz_om[4*iat+3][4*jat] = 0.0

                # Derivative of <p_i | p_j>
                stv = -8.0 * rcov[iat] * rcov[jat] * r * r * sji
                Dz_om[4*iat+1][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[0] * d[0] - 0.5/r) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[0] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dz_om[4*iat+1][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[1] * d[0]        ) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[1] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dz_om[4*iat+1][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[2] * d[0]        ) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[2] + d[2]) \
                                                                     * amp[iat] * amp[jat]
                Dz_om[4*iat+2][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[0] * d[1]        ) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[0] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dz_om[4*iat+2][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[1] * d[1] - 0.5/r) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[1] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dz_om[4*iat+2][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[2] * d[1]        ) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[2] + d[2]) \
                                                                     * amp[iat] * amp[jat]
                Dz_om[4*iat+3][4*jat+1] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[0] * d[2]        ) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[0] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dz_om[4*iat+3][4*jat+2] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[1] * d[2]        ) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[1] +  0.0) \
                                                                     * amp[iat] * amp[jat]
                Dz_om[4*iat+3][4*jat+3] = ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) \
                       * d[0] * stv * (d[2] * d[2] - 0.5/r) * amp[iat] * amp[jat] + \
                ( kron_delta(iat, D_n) - kron_delta(jat, D_n) ) * stv * (d[2] + d[2]) \
                                                                     * amp[iat] * amp[jat]
                # if D_n == 4*iat+1 and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dz_om[4*iat+1][4*jat+1] = -(4.0*r) * d[2] * stv \
                #     * (d[0] * d[0] - 0.5/r) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                #     Dz_om[4*iat+1][4*jat+2] = -(4.0*r) * d[2] * stv \
                #     * (d[1] * d[0]        ) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                #     Dz_om[4*iat+1][4*jat+3] = -(4.0*r) * d[2] * stv \
                #     * (d[2] * d[0]        ) * amp[iat] * amp[jat] + \
                #     stv * (1.0 * d[0]        ) * amp[iat] * amp[jat]
                # elif D_n == 4*iat+2 and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dz_om[4*iat+2][4*jat+1] = -(4.0*r) * d[2] * stv \
                #     * (d[0] * d[1]        ) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                #     Dz_om[4*iat+2][4*jat+2] = -(4.0*r) * d[2] * stv \
                #     * (d[1] * d[1] - 0.5/r) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                #     Dz_om[4*iat+2][4*jat+3] = -(4.0*r) * d[2] * stv \
                #     * (d[2] * d[1]        ) * amp[iat] * amp[jat] + \
                #     stv * (1.0 * d[1]        ) * amp[iat] * amp[jat]
                # elif D_n == 4*iat+3 and D_n != 4*jat and D_n != 4*jat+1 \
                # and D_n != 4*jat+2 and D_n != 4*jat+3:
                #     Dz_om[4*iat+3][4*jat+1] = -(4.0*r) * d[2] * stv \
                #     * (d[0] * d[2]        ) * amp[iat] * amp[jat] + \
                #     stv * (1.0 * d[0]        ) * amp[iat] * amp[jat]
                #     Dz_om[4*iat+3][4*jat+2] = -(4.0*r) * d[2] * stv \
                #     * (d[1] * d[2]        ) * amp[iat] * amp[jat] + \
                #     stv * (1.0 * d[1]        ) * amp[iat] * amp[jat]
                #     Dz_om[4*iat+3][4*jat+3] = -(4.0*r) * d[2] * stv \
                #     * (d[2] * d[2] - 0.5/r) * amp[iat] * amp[jat] + \
                #     stv * (2.0 * d[2]        ) * amp[iat] * amp[jat]
                # elif D_n == 4*jat+1 and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dz_om[4*iat+1][4*jat+1] = (4.0*r) * d[2] * stv \
                #     * (d[0] * d[0] - 0.5/r) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0          ) * amp[iat] * amp[jat]
                #     Dz_om[4*iat+2][4*jat+1] = (4.0*r) * d[2] * stv \
                #     * (d[1] * d[0]        ) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0          ) * amp[iat] * amp[jat]
                #     Dz_om[4*iat+3][4*jat+1] = (4.0*r) * d[2] * stv \
                #     * (d[2] * d[0]        ) * amp[iat] * amp[jat] + \
                #     stv * (-1.0 * d[0]        ) * amp[iat] * amp[jat]
                # elif D_n == 4*jat+2 and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dz_om[4*iat+1][4*jat+2] = (4.0*r) * d[2] * stv \
                #     * (d[0] * d[1]        ) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                #     Dz_om[4*iat+2][4*jat+2] = (4.0*r) * d[2] * stv \
                #     * (d[1] * d[1] - 0.5/r) * amp[iat] * amp[jat] + \
                #     stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                #     Dz_om[4*iat+3][4*jat+2] = (4.0*r) * d[2] * stv \
                #     * (d[2] * d[1]        ) * amp[iat] * amp[jat] + \
                #     stv * (-1.0 * d[1]        ) * amp[iat] * amp[jat]
                # elif D_n == 4*jat+3 and D_n != 4*iat and D_n != 4*iat+1 \
                # and D_n != 4*iat+2 and D_n != 4*iat+3:
                #     Dz_om[4*iat+1][4*jat+3] = (4.0*r) * d[2] * stv \
                #     * (d[0] * d[2]        ) * amp[iat] * amp[jat] + \
                #     stv * (-1.0 * d[0]        ) * amp[iat] * amp[jat]
                #     Dz_om[4*iat+2][4*jat+3] = (4.0*r) * d[2] * stv \
                #     * (d[1] * d[2]        ) * amp[iat] * amp[jat] + \
                #     stv * (-1.0 * d[1]       ) * amp[iat] * amp[jat]
                #     Dz_om[4*iat+3][4*jat+3] = (4.0*r) * d[2] * stv \
                #     * (d[2] * d[2] - 0.5/r) * amp[iat] * amp[jat] + \
                #     stv * (-2.0 * d[2]       ) * amp[iat] * amp[jat]
                # else:
                #     Dz_om[4*iat+1][4*jat+1] = 0.0
                #     Dz_om[4*iat+1][4*jat+2] = 0.0
                #     Dz_om[4*iat+1][4*jat+3] = 0.0
                #     Dz_om[4*iat+2][4*jat+1] = 0.0
                #     Dz_om[4*iat+2][4*jat+2] = 0.0
                #     Dz_om[4*iat+2][4*jat+3] = 0.0
                #     Dz_om[4*iat+3][4*jat+1] = 0.0
                #     Dz_om[4*iat+3][4*jat+2] = 0.0
                #     Dz_om[4*iat+3][4*jat+3] = 0.0
                
    return Dz_om



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
    if x == 0:
        Dx_om = get_Dx_gom(lseg, rxyz_sphere, rcov_sphere, amp, D_n)
        for i in range(N_vec):
            Dx_mul_V_om = np.matmul(Dx_om, sorted_Varr_om[:, i])
            D_fp[i][0] = np.matmul(sorted_Varr_om[:, i].T, Dx_mul_V_om)
    elif x == 1:
        Dy_om = get_Dy_gom(lseg, rxyz_sphere, rcov_sphere, amp, D_n)
        for j in range(N_vec):
            Dy_mul_V_om = np.matmul(Dy_om, sorted_Varr_om[:, j])
            D_fp[j][0] = np.matmul(sorted_Varr_om[:, j].T, Dy_mul_V_om)
    elif x == 2:
        Dz_om = get_Dz_gom(lseg, rxyz_sphere, rcov_sphere, amp, D_n)
        for k in range(N_vec):
            Dz_mul_V_om = np.matmul(Dz_om, sorted_Varr_om[:, k])
            D_fp[k][0] = np.matmul(sorted_Varr_om[:, k].T, Dz_mul_V_om)
    else:
        print("Error: Wrong x value! x can only be 0,1,2")
    
    # D_fp = np.real(D_fp)
    # print("D_fp {0:d} = {1:s}".format(x, np.array_str(D_fp, precision=6, suppress_small=False)) )
    D_fp_factor = np.zeros(N_vec)
    for N in range(N_vec):
        D_fp_factor[N] = 1/D_fp[N][0]
        D_fp[N][0] = (np.exp( np.log(D_fp_factor[N]*D_fp[N][0] + 1.2) ) - 1.2)/D_fp_factor[N]
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
    D_fp_mat = np.zeros((3, nx*lseg, nat)) + 1j*np.zeros((3, nx*lseg, nat))
    for i in range(3*nat):
        D_n = i // 3
        x = i % 3
        D_fp = get_D_fp(contract, ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, x, D_n, iat)
        for j in range(len(D_fp)):
            D_fp_mat[x][j][D_n] = D_fp[j][0]
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
    rxyz = rxyz.tolist()
    rxyz_sphere_j = rxyz_sphere_j.tolist()
    for j in range(nat_j_sphere):
        if rxyz[iat] == rxyz_sphere_j[j]:
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



# Previous CG process
'''
# @numba.jit()
def get_null_D_fp(lseg, rxyz, rcov, amp):
    D_fp_mat = get_D_fp_mat(lseg, rxyz, rcov, amp, D_n)
    null_D_fp = null_space(D_fp_mat)
    return null_D_fp


# @numba.jit()
def get_norm_fp(lseg, rxyz, rcov, amp):
    nat = len(rxyz)
    r_vec = rxyz.reshape(3*nat, 1)
    D_fp_mat = get_D_fp_mat(lseg, rxyz, rcov, amp)
    fp_vec = np.matmul(D_fp_mat, r_vec)
    norm_fp = np.linalg.norm(fp_vec)
    return norm_fp


# @numba.jit()
def get_grad_norm_fp(lseg, rxyz, rcov, amp):
    nat = len(rxyz)
    r_vec = rxyz.reshape(3*nat, 1)
    grad_norm_fp = zeros_like(r_vec)
    D_fp_mat = get_D_fp_mat(lseg, rxyz, rcov, amp)
    fp_vec = np.matmul(D_fp_mat, r_vec)
    norm_fp = get_norm_fp(lseg, rxyz, rcov, amp)
    for iat in range(3*nat):
        grad_norm_fp[iat] = np.matmul( np.transpose(D_fp_mat[:, iat]), r_vec ) / norm_fp
    return grad_norm_fp



# @numba.jit()
def get_CG_norm_fp(lseg, rxyz, rcov, amp):
    nat = len(rxyz)
    r_init = np.zeros_like( rxyz.reshape(3*nat, 1) )
    x0 = r_init
    f = get_norm_fp(lseg, rxyz, rcov, amp)
    gradf = get_grad_norm_fp(lseg, rxyz, rcov, amp)
    return minimize(f, x0, jac = gradf, method = 'CG', options = {'gtol': 1e-8, 'disp': True})


# @numba.jit()
def get_fpCG(contract, ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff):
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
    lfp = []
    sfp = []
    for iat in range(nat):
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
                            for il in range(lseg):
                                if il == 0:
                                    # print len(ind)
                                    # print ind
                                    # print il+lseg*(n_sphere-1)
                                    ind[il+lseg*(n_sphere-1)] = ityp_sphere * l
                                else:
                                    ind[il+lseg*(n_sphere-1)] = ityp_sphere * l + 1
                                    # ind[il+lseg*(n_sphere-1)] == ityp_sphere * l + 1
        n_sphere_list.append(n_sphere)
        rxyz_sphere = np.array(rxyz_sphere, float)
        # full overlap matrix
        # nid = lseg * n_sphere
        # om = get_gom(lseg, rxyz_sphere, rcov_sphere, amp)
        # val, vec = np.linalg.eig(om)
        # val = np.real(val)
        # fp0 = np.zeros(nx*lseg)
        # for i in range(len(val)):
        #     fp0[i] = val[i]
        # lfp.append(sorted(fp0))
        # pvec = np.real(np.transpose(vec)[0])

    print ("n_sphere_min", min(n_sphere_list))
    print ("n_shpere_max", max(n_sphere_list)) 

    res_CG = get_CG_norm_fp(lseg, rxyz_sphere, rcov_sphere, amp)
    
    return res_CG
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
                            for il in range(lseg):
                                if il == 0:
                                    # print len(ind)
                                    # print ind
                                    # print il+lseg*(n_sphere-1)
                                    ind[il+lseg*(n_sphere-1)] = ityp_sphere * l
                                else:
                                    ind[il+lseg*(n_sphere-1)] = ityp_sphere * l + 1
                                    # ind[il+lseg*(n_sphere-1)] == ityp_sphere * l + 1
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
    lfp = []
    sfp = []
    amp, n_sphere, rxyz_sphere, rcov_sphere = \
                   get_sphere(ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, iat)
    # full overlap matrix
    nid = lseg * n_sphere
    gom = get_gom(lseg, rxyz_sphere, rcov_sphere, amp)
    val, vec = np.linalg.eig(gom)
    val = np.real(val)
    fp0 = np.zeros(nx*lseg)
    for i in range(len(val)):
        fp0[i] = val[i]
    lfp = sorted(fp0)
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
        sfp0 = np.linalg.eigvals(omx)
        sfp.append(sorted(sfp0))

    # print ("n_sphere_min", min(n_sphere_list))
    # print ("n_shpere_max", max(n_sphere_list)) 

    if contract:
        sfp = np.array(sfp, float)
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
    lenfp = np.shape(fp1)
    # nat, lenfp = np.shape(fp1)
    # fpd = 0.0
    tfpd = fp1 - fp2
    # fpd = np.sqrt( np.dot(tfpd, tfpd)/lenfp )
    fpd = np.dot(tfpd, tfpd)/lenfp
    return fpd

'''
# @numba.jit()
def get_fpdist(ntyp, types, fp1, fp2):
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
                        MX[iat][jat] = np.sqrt(np.vdot(tfpd, tfpd)/lenfp)

        row_ind, col_ind = linear_sum_assignment(MX)
        # print(row_ind, col_ind)
        total = MX[row_ind, col_ind].sum()
        fpd += total

    fpd = fpd / nat
    return fpd
'''


