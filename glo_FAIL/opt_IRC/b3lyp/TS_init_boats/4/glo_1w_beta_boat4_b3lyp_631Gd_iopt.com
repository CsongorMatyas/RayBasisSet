%NPROCS=4
# opt=(calcfc,ts,noeigentest,maxcycle=1000,cartesian) b3lyp 6-31g(d) freq

 ./4/glo_1w_beta_boat4_b3lyp_631Gd_iopt.out

0  1
O          -1.54419         0.24263         0.33130
C           0.84051         0.54339        -0.35787
C          -0.30895        -1.59526        -0.79884
C           0.86729        -0.95435        -0.05551
C          -1.61797        -0.80158        -0.64250
C          -0.43377         1.15408         0.26567
H           0.71945        -1.10176         1.02723
H          -0.19010         1.37329         1.31313
H           0.83963         0.67637        -1.44926
H          -2.39730        -1.46052        -0.24649
C          -0.83918         2.45687        -0.43343
H          -1.02728         2.26717        -1.49328
H          -1.75641         2.86287         0.01296
H          -0.05025        -1.59523        -1.86765
O          -1.95243        -0.28217        -1.90852
H          -2.83594         0.11455        -1.83686
O           0.23560         3.40869        -0.37517
H           0.20791         3.83354         0.49670
O          -0.53264        -2.91631        -0.33944
H           0.33885        -3.34703        -0.37802
O           2.01320        -1.63664        -0.51416
H           2.77794        -1.39843         0.06110
O           1.97904         1.21554         0.19267
H           1.88137         2.14778        -0.07840
O          -2.98133        -0.59422         2.70574
H          -3.82569        -0.14839         2.54667
H          -2.41421        -0.27939         1.97898
O           3.96314        -0.49743         1.06351
H           3.76803        -0.59205         2.00818
H           3.47322         0.30773         0.79235

