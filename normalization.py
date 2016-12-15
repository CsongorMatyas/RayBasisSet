import numpy as np
import sys

AV = [750.0779372, 137.5259687, 38.4748101, 13.21713568, 5.132600557, 2.11388835]
DV = [0.9163596, 0.049361493, 0.168538305, 0.3705628, 0.41649153, 0.130334084]

def normalization(DV,AV):
    lenDV = len(DV)
    lenAV = len(AV)
    if lenDV != lenAV:
        print('AV != DV')
        sys.exit()
    D_product = np.asarray([dv_i*dv_j for dv_i in DV for dv_j in DV])
    A_product = np.asarray([(a_i*a_j)/(a_i+a_j)**2 for a_i in AV for a_j in AV])
    norm = np.dot(D_product, np.power(A_product,0.75)) * 2.0 * np.sqrt(2.0)
    Corr_DV = DV / np.sqrt(norm)
    return norm, Corr_DV

norm, Corr_DV =  normalization(DV,AV)
print(DV)
print(Corr_DV)
print(norm)
norm, Corr_DV =  normalization(Corr_DV,AV)
print(norm)
