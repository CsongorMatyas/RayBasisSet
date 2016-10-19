#!/usr/bin/env python

import numpy as np
from scipy.optimize import minimize

def Function(Values):
        X=Values[0]
        Y=Values[1]
        Z=Values[2]
        V=Values[3]
        W=Values[4]
        equation = round(X**4, 15) + round(Y**2, 15) + round(Z**2, 15) + round(V**2, 15) + round(Z**2, 15)
        return round(equation, 15)



x0 = np.array([100.0, -100, 54, 72, -9])
result = minimize(Function, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})

print(result.x)