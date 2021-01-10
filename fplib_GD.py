import numpy as np
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
                    * np.exp(-1*d2*r) * amp[iat] * amp[jat]
                
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
                if D_n == iat:
                    Dx_om[iat][jat] = -(4.0*r) * d[0] * sji * amp[iat] * amp[jat]
                else if D_n == jat:
                    Dx_om[iat][jat] =  (4.0*r) * d[0] * sji * amp[iat] * amp[jat]
                else:
                    Dx_om[iat][jat] = 0.0
                
    else:
        # for both s and p orbitals
        Dx_om = np.zeros((4*nat, 4*nat))
        for iat in range(nat):
            for jat in range(nat):
                d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                sji = np.sqrt(4.0*rcov[iat]*rcov[jat])**3 * np.exp(-1*d2*r)
                # Derivative of <s_i | s_j>
                if D_n == 4*iat and D_n != 4*jat and D_n != 4*jat+1 and  \
                D_n != 4*jat+2 and D_n != 4*jat:
                    Dx_om[4*iat][4*jat] = -(4.0*r) * d[0] * sji * amp[iat] * amp[jat]
                else if D_n == 4*jat and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dx_om[4*iat][4*jat] =  (4.0*r) * d[0] * sji * amp[iat] * amp[jat]
                else:
                    Dx_om[4*iat][4*jat] = 0.0
                
                # Derivative of <s_i | p_j>
                stv = np.sqrt(8.0) * rcov[jat] * r * sji
                if D_n == 4*jat+1 and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dx_om[4*iat][4*jat+1] = stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[0], d[0] ) * 4.0*r * amp[iat] * amp[jat] 
                else if D_n == 4*jat+2 and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dx_om[4*iat][4*jat+2] = stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[0], d[0] ) * 4.0*r * amp[iat] * amp[jat] 
                else if D_n == 4*jat+3 and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dx_om[4*iat][4*jat+3] = stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[0], d[0] ) * 4.0*r * amp[iat] * amp[jat] 
                else if D_n == 4*iat and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dx_om[4*iat][4*jat+1] = - stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[0], d[0] ) * 4.0*r * amp[iat] * amp[jat] 
                    Dx_om[4*iat][4*jat+2] = - stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[1], d[1] ) * 4.0*r * amp[iat] * amp[jat] 
                    Dx_om[4*iat][4*jat+3] = - stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                else:
                    Dx_om[4*iat][4*jat+1] = 0.0
                    Dx_om[4*iat][4*jat+2] = 0.0
                    Dx_om[4*iat][4*jat+3] = 0.0

                # Derivative of <p_i | s_j>
                stv = np.sqrt(8.0) * rcov[iat] * r * sji * -1.0
                if D_n == 4*iat+1 and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dx_om[4*iat+1][4*jat] = stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[0], d[0] ) * 4.0*r * amp[iat] * amp[jat] 
                else if D_n == 4*iat+2 and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dx_om[4*iat+2][4*jat] = stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[0], d[0] ) * 4.0*r * amp[iat] * amp[jat] 
                else if D_n == 4*iat+3 and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dx_om[4*iat+3][4*jat] = stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[0], d[0] ) * 4.0*r * amp[iat] * amp[jat] 
                else if D_n == 4*jat and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dx_om[4*iat+1][4*jat] = - stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[0], d[0] ) * 4.0*r * amp[iat] * amp[jat] 
                    Dx_om[4*iat+2][4*jat] = - stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[1], d[1] ) * 4.0*r * amp[iat] * amp[jat] 
                    Dx_om[4*iat+3][4*jat] = - stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                else:
                    Dx_om[4*iat+1][4*jat] = 0.0
                    Dx_om[4*iat+2][4*jat] = 0.0
                    Dx_om[4*iat+3][4*jat] = 0.0

                # Derivative of <p_i | p_j>
                stv = -8.0 * rcov[iat] * rcov[jat] * r * r * sji
                if D_n == 4*iat+1 and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dx_om[4*iat+1][4*jat+1] = -(4.0*r) * d[0] * stv \
                    * (d[0] * d[0] - 0.5/r) * amp[iat] * amp[jat] + \
                    stv * (2.0 * d[0]        ) * amp[iat] * amp[jat]
                    Dx_om[4*iat+1][4*jat+2] = -(4.0*r) * d[0] * stv \
                    * (d[1] * d[0]        ) * amp[iat] * amp[jat] + \
                    stv * (1.0 * d[1]        ) * amp[iat] * amp[jat]
                    Dx_om[4*iat+1][4*jat+3] = -(4.0*r) * d[0] * stv \
                    * (d[2] * d[0]        ) * amp[iat] * amp[jat] + \
                    stv * (1.0 * d[2]        ) * amp[iat] * amp[jat]
                else if D_n == 4*iat+2 and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dx_om[4*iat+2][4*jat+1] = -(4.0*r) * d[0] * stv \
                    * (d[0] * d[1]        ) * amp[iat] * amp[jat] + \
                    stv * (1.0 * d[1]        ) * amp[iat] * amp[jat]
                    Dx_om[4*iat+2][4*jat+2] = -(4.0*r) * d[0] * stv \
                    * (d[1] * d[1] - 0.5/r) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0        ) * amp[iat] * amp[jat]
                    Dx_om[4*iat+2][4*jat+3] = -(4.0*r) * d[0] * stv \
                    * (d[2] * d[1]        ) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0        ) * amp[iat] * amp[jat]
                else if D_n == 4*iat+3 and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dx_om[4*iat+3][4*jat+1] = -(4.0*r) * d[0] * stv \
                    * (d[0] * d[2]        ) * amp[iat] * amp[jat] + \
                    stv * (1.0 * d[2]        ) * amp[iat] * amp[jat]
                    Dx_om[4*iat+3][4*jat+2] = -(4.0*r) * d[0] * stv \
                    * (d[1] * d[2]        ) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0        ) * amp[iat] * amp[jat]
                    Dx_om[4*iat+3][4*jat+3] = -(4.0*r) * d[0] * stv \
                    * (d[2] * d[2] - 0.5/r) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0        ) * amp[iat] * amp[jat]
                else if D_n == 4*jat+1 and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dx_om[4*iat+1][4*jat+1] = (4.0*r) * d[0] * stv \
                    * (d[0] * d[0] - 0.5/r) * amp[iat] * amp[jat] + \
                    stv * (-2.0 * d[0]        ) * amp[iat] * amp[jat]
                    Dx_om[4*iat+2][4*jat+1] = (4.0*r) * d[0] * stv \
                    * (d[1] * d[0]        ) * amp[iat] * amp[jat] + \
                    stv * (-1.0 * d[1]        ) * amp[iat] * amp[jat]
                    Dx_om[4*iat+3][4*jat+1] = (4.0*r) * d[0] * stv \
                    * (d[2] * d[0]        ) * amp[iat] * amp[jat] + \
                    stv * (-1.0 * d[2]        ) * amp[iat] * amp[jat]
                else if D_n == 4*jat+2 and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dx_om[4*iat+1][4*jat+2] = (4.0*r) * d[0] * stv \
                    * (d[0] * d[1]        ) * amp[iat] * amp[jat] + \
                    stv * (-1.0 * d[1]        ) * amp[iat] * amp[jat]
                    Dx_om[4*iat+2][4*jat+2] = (4.0*r) * d[0] * stv \
                    * (d[1] * d[1] - 0.5/r) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0        ) * amp[iat] * amp[jat]
                    Dx_om[4*iat+3][4*jat+2] = (4.0*r) * d[0] * stv \
                    * (d[2] * d[1]        ) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0        ) * amp[iat] * amp[jat]
                else if D_n == 4*jat+3 and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dx_om[4*iat+1][4*jat+3] = (4.0*r) * d[0] * stv \
                    * (d[0] * d[2]        ) * amp[iat] * amp[jat] + \
                    stv * (-1.0 * d[2]        ) * amp[iat] * amp[jat]
                    Dx_om[4*iat+2][4*jat+3] = (4.0*r) * d[0] * stv \
                    * (d[1] * d[2]        ) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0        ) * amp[iat] * amp[jat]
                    Dx_om[4*iat+3][4*jat+3] = (4.0*r) * d[0] * stv \
                    * (d[2] * d[2] - 0.5/r) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0        ) * amp[iat] * amp[jat]
                else:
                    Dx_om[4*iat+1][4*jat+1] = 0.0
                    Dx_om[4*iat+1][4*jat+2] = 0.0
                    Dx_om[4*iat+1][4*jat+3] = 0.0
                    Dx_om[4*iat+2][4*jat+1] = 0.0
                    Dx_om[4*iat+2][4*jat+2] = 0.0
                    Dx_om[4*iat+2][4*jat+3] = 0.0
                    Dx_om[4*iat+3][4*jat+1] = 0.0
                    Dx_om[4*iat+3][4*jat+2] = 0.0
                    Dx_om[4*iat+3][4*jat+3] = 0.0
                
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
                if D_n == iat:
                    Dy_om[iat][jat] = -(4.0*r) * d[1] * sji * amp[iat] * amp[jat]
                else if D_n == jat:
                    Dy_om[iat][jat] =  (4.0*r) * d[1] * sji * amp[iat] * amp[jat]
                else:
                    Dy_om[iat][jat] = 0.0
                
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
                if D_n == 4*iat and D_n != 4*jat and D_n != 4*jat+1 and  \
                D_n != 4*jat+2 and D_n != 4*jat:
                    Dy_om[4*iat][4*jat] = -(4.0*r) * d[1] * sji * amp[iat] * amp[jat]
                else if D_n == 4*jat and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dy_om[4*iat][4*jat] =  (4.0*r) * d[1] * sji * amp[iat] * amp[jat]
                else:
                    Dy_om[4*iat][4*jat] = 0.0
                
                # Derivative of <s_i | p_j>
                stv = np.sqrt(8.0) * rcov[jat] * r * sji
                if D_n == 4*jat+1 and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dy_om[4*iat][4*jat+1] = stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[1], d[1] ) * 4.0*r * amp[iat] * amp[jat] 
                else if D_n == 4*jat+2 and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dy_om[4*iat][4*jat+2] = stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[1], d[1] ) * 4.0*r * amp[iat] * amp[jat] 
                else if D_n == 4*jat+3 and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dy_om[4*iat][4*jat+3] = stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[1], d[1] ) * 4.0*r * amp[iat] * amp[jat] 
                else if D_n == 4*iat and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dy_om[4*iat][4*jat+1] = - stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[0], d[0] ) * 4.0*r * amp[iat] * amp[jat] 
                    Dy_om[4*iat][4*jat+2] = - stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[1], d[1] ) * 4.0*r * amp[iat] * amp[jat] 
                    Dy_om[4*iat][4*jat+3] = - stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                else:
                    Dy_om[4*iat][4*jat+1] = 0.0
                    Dy_om[4*iat][4*jat+2] = 0.0
                    Dy_om[4*iat][4*jat+3] = 0.0

                # Derivative of <p_i | s_j>
                stv = np.sqrt(8.0) * rcov[iat] * r * sji * -1.0
                if D_n == 4*iat+1 and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dy_om[4*iat+1][4*jat] = stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[1], d[1] ) * 4.0*r * amp[iat] * amp[jat] 
                else if D_n == 4*iat+2 and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dy_om[4*iat+2][4*jat] = stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[1], d[1] ) * 4.0*r * amp[iat] * amp[jat] 
                else if D_n == 4*iat+3 and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dy_om[4*iat+3][4*jat] = stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[1], d[1] ) * 4.0*r * amp[iat] * amp[jat] 
                else if D_n == 4*jat and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dy_om[4*iat+1][4*jat] = - stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[0], d[0] ) * 4.0*r * amp[iat] * amp[jat] 
                    Dy_om[4*iat+2][4*jat] = - stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[1], d[1] ) * 4.0*r * amp[iat] * amp[jat] 
                    Dy_om[4*iat+3][4*jat] = - stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                else:
                    Dy_om[4*iat+1][4*jat] = 0.0
                    Dy_om[4*iat+2][4*jat] = 0.0
                    Dy_om[4*iat+3][4*jat] = 0.0

                # Derivative of <p_i | p_j>
                stv = -8.0 * rcov[iat] * rcov[jat] * r * r * sji
                if D_n == 4*iat+1 and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dy_om[4*iat+1][4*jat+1] = -(4.0*r) * d[1] * stv \
                    * (d[0] * d[0] - 0.5/r) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                    Dx_om[4*iat+1][4*jat+2] = -(4.0*r) * d[1] * stv \
                    * (d[1] * d[0]        ) * amp[iat] * amp[jat] + \
                    stv * (1.0 * d[0]        ) * amp[iat] * amp[jat]
                    Dx_om[4*iat+1][4*jat+3] = -(4.0*r) * d[1] * stv \
                    * (d[2] * d[0]        ) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                else if D_n == 4*iat+2 and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dy_om[4*iat+2][4*jat+1] = -(4.0*r) * d[1] * stv \
                    * (d[0] * d[1]        ) * amp[iat] * amp[jat] + \
                    stv * (1.0 * d[0]        ) * amp[iat] * amp[jat]
                    Dy_om[4*iat+2][4*jat+2] = -(4.0*r) * d[1] * stv \
                    * (d[1] * d[1] - 0.5/r) * amp[iat] * amp[jat] + \
                    stv * (2.0 * d[1]        ) * amp[iat] * amp[jat]
                    Dy_om[4*iat+2][4*jat+3] = -(4.0*r) * d[1] * stv \
                    * (d[2] * d[1]        ) * amp[iat] * amp[jat] + \
                    stv * (1.0 * d[2]        ) * amp[iat] * amp[jat]
                else if D_n == 4*iat+3 and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dy_om[4*iat+3][4*jat+1] = -(4.0*r) * d[1] * stv \
                    * (d[0] * d[2]        ) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                    Dy_om[4*iat+3][4*jat+2] = -(4.0*r) * d[1] * stv \
                    * (d[1] * d[2]        ) * amp[iat] * amp[jat] + \
                    stv * (1.0 * d[2]       ) * amp[iat] * amp[jat]
                    Dy_om[4*iat+3][4*jat+3] = -(4.0*r) * d[1] * stv \
                    * (d[2] * d[2] - 0.5/r) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                else if D_n == 4*jat+1 and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dy_om[4*iat+1][4*jat+1] = (4.0*r) * d[1] * stv \
                    * (d[0] * d[0] - 0.5/r) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0          ) * amp[iat] * amp[jat]
                    Dy_om[4*iat+2][4*jat+1] = (4.0*r) * d[1] * stv \
                    * (d[1] * d[0]        ) * amp[iat] * amp[jat] + \
                    stv * (-1.0 * d[0]        ) * amp[iat] * amp[jat]
                    Dy_om[4*iat+3][4*jat+1] = (4.0*r) * d[1] * stv \
                    * (d[2] * d[0]        ) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0          ) * amp[iat] * amp[jat]
                else if D_n == 4*jat+2 and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dy_om[4*iat+1][4*jat+2] = (4.0*r) * d[1] * stv \
                    * (d[0] * d[1]        ) * amp[iat] * amp[jat] + \
                    stv * (-1.0 * d[0]        ) * amp[iat] * amp[jat]
                    Dy_om[4*iat+2][4*jat+2] = (4.0*r) * d[1] * stv \
                    * (d[1] * d[1] - 0.5/r) * amp[iat] * amp[jat] + \
                    stv * (-2.0 * d[1]       ) * amp[iat] * amp[jat]
                    Dy_om[4*iat+3][4*jat+2] = (4.0*r) * d[1] * stv \
                    * (d[2] * d[1]        ) * amp[iat] * amp[jat] + \
                    stv * (-1.0 * d[2]        ) * amp[iat] * amp[jat]
                else if D_n == 4*jat+3 and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dy_om[4*iat+1][4*jat+3] = (4.0*r) * d[1] * stv \
                    * (d[0] * d[2]        ) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0          ) * amp[iat] * amp[jat]
                    Dy_om[4*iat+2][4*jat+3] = (4.0*r) * d[1] * stv \
                    * (d[1] * d[2]        ) * amp[iat] * amp[jat] + \
                    stv * (-1.0 * d[2]        ) * amp[iat] * amp[jat]
                    Dy_om[4*iat+3][4*jat+3] = (4.0*r) * d[1] * stv \
                    * (d[2] * d[2] - 0.5/r) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                else:
                    Dy_om[4*iat+1][4*jat+1] = 0.0
                    Dy_om[4*iat+1][4*jat+2] = 0.0
                    Dy_om[4*iat+1][4*jat+3] = 0.0
                    Dy_om[4*iat+2][4*jat+1] = 0.0
                    Dy_om[4*iat+2][4*jat+2] = 0.0
                    Dy_om[4*iat+2][4*jat+3] = 0.0
                    Dy_om[4*iat+3][4*jat+1] = 0.0
                    Dy_om[4*iat+3][4*jat+2] = 0.0
                    Dy_om[4*iat+3][4*jat+3] = 0.0
                
    return Dy_om

# @numba.jit()
def get_Dx_gom(lseg, rxyz, rcov, amp, D_n):
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
                if D_n == iat:
                    Dz_om[iat][jat] = -(4.0*r) * d[2] * sji * amp[iat] * amp[jat]
                else if D_n == jat:
                    Dz_om[iat][jat] =  (4.0*r) * d[2] * sji * amp[iat] * amp[jat]
                else:
                    Dz_om[iat][jat] = 0.0
                
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
                if D_n == 4*iat and D_n != 4*jat and D_n != 4*jat+1 and  \
                D_n != 4*jat+2 and D_n != 4*jat:
                    Dz_om[4*iat][4*jat] = -(4.0*r) * d[2] * sji * amp[iat] * amp[jat]
                else if D_n == 4*jat and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dz_om[4*iat][4*jat] =  (4.0*r) * d[2] * sji * amp[iat] * amp[jat]
                else:
                    Dz_om[4*iat][4*jat] = 0.0
                
                # Derivative of <s_i | p_j>
                stv = np.sqrt(8.0) * rcov[jat] * r * sji
                if D_n == 4*jat+1 and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dz_om[4*iat][4*jat+1] = stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                else if D_n == 4*jat+2 and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dz_om[4*iat][4*jat+2] = stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                else if D_n == 4*jat+3 and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dz_om[4*iat][4*jat+3] = stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                else if D_n == 4*iat and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dz_om[4*iat][4*jat+1] = - stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[0], d[0] ) * 4.0*r * amp[iat] * amp[jat] 
                    Dz_om[4*iat][4*jat+2] = - stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[1], d[1] ) * 4.0*r * amp[iat] * amp[jat] 
                    Dz_om[4*iat][4*jat+3] = - stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                else:
                    Dz_om[4*iat][4*jat+1] = 0.0
                    Dz_om[4*iat][4*jat+2] = 0.0
                    Dz_om[4*iat][4*jat+3] = 0.0

                # Derivative of <p_i | s_j>
                stv = np.sqrt(8.0) * rcov[iat] * r * sji * -1.0
                if D_n == 4*iat+1 and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dz_om[4*iat+1][4*jat] = stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                else if D_n == 4*iat+2 and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dz_om[4*iat+2][4*jat] = stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                else if D_n == 4*iat+3 and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dz_om[4*iat+3][4*jat] = stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                else if D_n == 4*jat and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dz_om[4*iat+1][4*jat] = - stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[0], d[0] ) * 4.0*r * amp[iat] * amp[jat] 
                    Dz_om[4*iat+2][4*jat] = - stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[1], d[1] ) * 4.0*r * amp[iat] * amp[jat] 
                    Dz_om[4*iat+3][4*jat] = - stv * amp[iat] * amp[jat] \
                    + stv * np.dot( d[2], d[2] ) * 4.0*r * amp[iat] * amp[jat] 
                else:
                    Dz_om[4*iat+1][4*jat] = 0.0
                    Dz_om[4*iat+2][4*jat] = 0.0
                    Dz_om[4*iat+3][4*jat] = 0.0

                # Derivative of <p_i | p_j>
                stv = -8.0 * rcov[iat] * rcov[jat] * r * r * sji
                if D_n == 4*iat+1 and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dz_om[4*iat+1][4*jat+1] = -(4.0*r) * d[2] * stv \
                    * (d[0] * d[0] - 0.5/r) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                    Dz_om[4*iat+1][4*jat+2] = -(4.0*r) * d[2] * stv \
                    * (d[1] * d[0]        ) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                    Dz_om[4*iat+1][4*jat+3] = -(4.0*r) * d[2] * stv \
                    * (d[2] * d[0]        ) * amp[iat] * amp[jat] + \
                    stv * (1.0 * d[0]        ) * amp[iat] * amp[jat]
                else if D_n == 4*iat+2 and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dz_om[4*iat+2][4*jat+1] = -(4.0*r) * d[2] * stv \
                    * (d[0] * d[1]        ) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                    Dz_om[4*iat+2][4*jat+2] = -(4.0*r) * d[2] * stv \
                    * (d[1] * d[1] - 0.5/r) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                    Dz_om[4*iat+2][4*jat+3] = -(4.0*r) * d[2] * stv \
                    * (d[2] * d[1]        ) * amp[iat] * amp[jat] + \
                    stv * (1.0 * d[1]        ) * amp[iat] * amp[jat]
                else if D_n == 4*iat+3 and D_n != 4*jat and D_n != 4*jat+1 \
                and D_n != 4*jat+2 and D_n != 4*jat+3:
                    Dz_om[4*iat+3][4*jat+1] = -(4.0*r) * d[2] * stv \
                    * (d[0] * d[2]        ) * amp[iat] * amp[jat] + \
                    stv * (1.0 * d[0]        ) * amp[iat] * amp[jat]
                    Dz_om[4*iat+3][4*jat+2] = -(4.0*r) * d[2] * stv \
                    * (d[1] * d[2]        ) * amp[iat] * amp[jat] + \
                    stv * (1.0 * d[1]        ) * amp[iat] * amp[jat]
                    Dz_om[4*iat+3][4*jat+3] = -(4.0*r) * d[2] * stv \
                    * (d[2] * d[2] - 0.5/r) * amp[iat] * amp[jat] + \
                    stv * (2.0 * d[2]        ) * amp[iat] * amp[jat]
                else if D_n == 4*jat+1 and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dz_om[4*iat+1][4*jat+1] = (4.0*r) * d[2] * stv \
                    * (d[0] * d[0] - 0.5/r) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0          ) * amp[iat] * amp[jat]
                    Dz_om[4*iat+2][4*jat+1] = (4.0*r) * d[2] * stv \
                    * (d[1] * d[0]        ) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0          ) * amp[iat] * amp[jat]
                    Dz_om[4*iat+3][4*jat+1] = (4.0*r) * d[2] * stv \
                    * (d[2] * d[0]        ) * amp[iat] * amp[jat] + \
                    stv * (-1.0 * d[0]        ) * amp[iat] * amp[jat]
                else if D_n == 4*jat+2 and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dz_om[4*iat+1][4*jat+2] = (4.0*r) * d[2] * stv \
                    * (d[0] * d[1]        ) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                    Dz_om[4*iat+2][4*jat+2] = (4.0*r) * d[2] * stv \
                    * (d[1] * d[1] - 0.5/r) * amp[iat] * amp[jat] + \
                    stv * (0.0 * 0.0         ) * amp[iat] * amp[jat]
                    Dz_om[4*iat+3][4*jat+2] = (4.0*r) * d[2] * stv \
                    * (d[2] * d[1]        ) * amp[iat] * amp[jat] + \
                    stv * (-1.0 * d[1]        ) * amp[iat] * amp[jat]
                else if D_n == 4*jat+3 and D_n != 4*iat and D_n != 4*iat+1 \
                and D_n != 4*iat+2 and D_n != 4*iat+3:
                    Dz_om[4*iat+1][4*jat+3] = (4.0*r) * d[2] * stv \
                    * (d[0] * d[2]        ) * amp[iat] * amp[jat] + \
                    stv * (-1.0 * d[0]        ) * amp[iat] * amp[jat]
                    Dz_om[4*iat+2][4*jat+3] = (4.0*r) * d[2] * stv \
                    * (d[1] * d[2]        ) * amp[iat] * amp[jat] + \
                    stv * (-1.0 * d[1]       ) * amp[iat] * amp[jat]
                    Dz_om[4*iat+3][4*jat+3] = (4.0*r) * d[2] * stv \
                    * (d[2] * d[2] - 0.5/r) * amp[iat] * amp[jat] + \
                    stv * (-2.0 * d[2]       ) * amp[iat] * amp[jat]
                else:
                    Dz_om[4*iat+1][4*jat+1] = 0.0
                    Dz_om[4*iat+1][4*jat+2] = 0.0
                    Dz_om[4*iat+1][4*jat+3] = 0.0
                    Dz_om[4*iat+2][4*jat+1] = 0.0
                    Dz_om[4*iat+2][4*jat+2] = 0.0
                    Dz_om[4*iat+2][4*jat+3] = 0.0
                    Dz_om[4*iat+3][4*jat+1] = 0.0
                    Dz_om[4*iat+3][4*jat+2] = 0.0
                    Dz_om[4*iat+3][4*jat+3] = 0.0
                
    return Dz_om

# @numba.jit()
def get_D_fp(lseg, rxyz, rcov, amp, x, D_n):
    om = get_gom(lseg, rxyz, rcov, amp)
    lamda_om, Varr_om = np.linalg.eig(om)
    V_om = Varr_om[:, D_n-1]
    if x==0:
        Dx_om = get_Dx_gom(lseg, rxyz, rcov, amp, D_n)
        Dx_mul_V_om = np.matmul(Dx_om, V_om)
        D_fp = np.matmul(V_om.T, Dx_mul_V_om)
    else if x==1:
        Dy_om = get_Dy_gom(lseg, rxyz, rcov, amp, D_n)
        Dy_mul_V_om = np.matmul(Dy_om, V_om)
        D_fp = np.matmul(V_om.T, Dy_mul_V_om)
    else if x==2:
        Dz_om = get_Dz_gom(lseg, rxyz, rcov, amp, D_n)
        Dz_mul_V_om = np.matmul(Dz_om, V_om)
        D_fp = np.matmul(V_om.T, Dz_mul_V_om)
    else:
        print("Error: Wrong x value! x can only be 0,1,2")
    return D_fp

# @numba.jit()
def get_D_fp_mat(lseg, rxyz, rcov, amp):
    om = get_gom(lseg, rxyz, rcov, amp)
    lamda_om, Varr_om = np.linalg.eig(om)
    N = len(lamda_om)
    nat = len(rxyz)
    for i in range(N):
        for j in range(3*nat):
            D_n = j
            x = j % 3
            D_fp_mat[i][j]=get_D_fp(lseg, rxyz, rcov, amp, x, D_n)
    return  D_fp_mat



# @numba.jit()
def get_null_D_fp(lseg, rxyz, rcov, amp):
    D_fp_mat = get_D_fp_mat(lseg, rxyz, rcov, amp)
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
    nat=len(rxyz)
    r_vec=rxyz.reshape(3*nat, 1)
    grad_norm_fp = zeros_like(r_vec)
    D_fp_mat = get_D_fp_mat(lseg, rxyz, rcov, amp)
    fp_vec=np.matmul(D_fp_mat, r_vec)
    norm_fp = get_norm_fp(lseg, rxyz, rcov, amp)
    for i in range(3*nat):
        grad_norm_fp[i] = np.matmul( np.transpose(D_fp_mat[:, i]), r_vec ) / norm_fp
    return grad_norm_fp



# @numba.jit()
def get_CG_norm_fp(lseg, rxyz, rcov, amp):
    r_init = np.zeros_like( rxyz.reshape(3*nat, 1) )
    x0 = r_init
    f = get_norm_fp(lseg, rxyz, rcov, amp)
    gradf = get_grad_norm_fp(lseg, rxyz, rcov, amp)
    return minimize(f, x0, jac=gradf, method='CG', options={'gtol': 1e-8, 'disp': True})



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
def get_fp(contract, ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff):
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
                                    ind[il+lseg*(n_sphere-1)] == ityp_sphere * l + 1
        n_sphere_list.append(n_sphere)
        rxyz_sphere = np.array(rxyz_sphere, float)
        # full overlap matrix
        nid = lseg * n_sphere
        gom = get_gom(lseg, rxyz_sphere, rcov_sphere, amp)
        val, vec = np.linalg.eig(gom)
        val = np.real(val)
        fp0 = np.zeros(nx*lseg)
        for i in range(len(val)):
            fp0[i] = val[i]
        lfp.append(sorted(fp0))
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


    print ("n_sphere_min", min(n_sphere_list))
    print ("n_shpere_max", max(n_sphere_list)) 

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
    vec = np.linalg.eigvals(lat2)
    # print (vec)
    ixyz = int(np.sqrt(1.0/max(vec))*cutoff) + 1
    return ixyz

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

