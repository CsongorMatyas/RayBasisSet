%NPROCS=4
# opt=(calcfc,ts,noeigentest,maxcycle=1000,cartesian) b3lyp 6-31g(d) freq

 ./2/glo_2w_beta_boat2_b3lyp_631Gd_iopt.out

0  1
C          -0.26192        -0.93619        -1.05534
O           0.49394         0.19506        -0.57884
H           2.96197         0.65820        -0.19841
H           1.66997        -2.02139         1.92576
O           2.01950        -2.21556         1.04358
H           2.62890        -1.47289         0.79816
O           3.32828        -0.24294        -0.31420
H           2.77383        -0.59163        -1.03329
C          -1.51881         1.19278         0.47750
C          -0.24364         1.40864        -0.39629
H          -1.52388         1.92218         1.29554
H          -0.56492         1.78911        -1.37952
O          -2.72225         1.40503        -0.24074
H          -2.90629         0.58657        -0.74094
C           0.72508         2.38818         0.27262
O           2.02203         2.41941        -0.31823
H           0.88894         2.08301         1.31024
H           0.26855         3.38854         0.28512
H           1.92924         2.65441        -1.25555
H           1.03588        -2.22036        -0.42565
C          -1.37759        -1.28281        -0.02692
H          -1.15857        -2.26070         0.41893
C          -1.44887        -0.22211         1.09287
H          -2.34694        -0.38673         1.69878
O          -2.60248        -1.33815        -0.77828
H          -3.28896        -1.72745        -0.21280
O          -0.33582        -0.35315         1.97262
H           0.44713        -0.31194         1.39347
O           0.63121        -1.95768        -1.29277
H          -0.73569        -0.67608        -2.01168

