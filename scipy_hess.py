#!/usr/bin/env python

import numpy as np
from scipy.optimize import minimize

def EquValue2(scalevalues):
        X=scalevalues[0]
        Y=scalevalues[1]
        Z=scalevalues[2]
        Y1=scalevalues[3]
        Y2=scalevalues[4]
        equ=round(X**4, 15)+round(Y**2, 15) + round(Z**2, 15)+round(Y1**2, 15)+round(Y2**2, 15)
        return round(equ, 15)

x0 = np.array([100.0, -100, 54, 72, -9])
res = minimize(EquValue2, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})

print(res.x)