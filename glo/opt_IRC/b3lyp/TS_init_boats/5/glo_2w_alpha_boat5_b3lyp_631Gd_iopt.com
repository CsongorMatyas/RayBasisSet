%NPROCS=4
# opt=(calcfc,ts,noeigentest,maxcycle=1000,cartesian) b3lyp 6-31g(d) freq

 ./5/glo_2w_alpha_boat5_b3lyp_631Gd_iopt.out

0  1
O           0.47949        -0.66229        -0.80210
C          -1.07441        -0.22878         1.08304
C          -0.48954         1.59937        -0.65410
C          -1.38780         1.19361         0.54112
C           0.77270         0.73693        -0.73890
C          -0.64702        -1.18014        -0.03706
H          -1.20873         1.89761         1.36575
H          -0.27269        -2.09694         0.43888
H          -1.98869        -0.64391         1.52138
H          -1.07578         1.44825        -1.56513
C          -1.77136        -1.56654        -1.01080
H          -1.33223        -2.12612        -1.84958
H          -2.26171        -0.67663        -1.40915
O          -2.79135        -2.30677        -0.35047
H          -2.42828        -3.17712        -0.12291
O          -0.12235         2.96510        -0.63864
H           0.59168         3.03422         0.01941
O          -2.72658         1.30164         0.06212
H          -3.32841         1.11764         0.80084
O          -0.11276        -0.20413         2.12703
H           0.66889         0.26640         1.77033
H           1.31269         0.94262        -1.66895
O           1.57523         1.07875         0.36866
H           3.73712        -0.83681        -0.03781
O           2.66016        -2.21888         0.02437
H           2.42347        -2.30500         0.96156
H           1.88876        -1.75403        -0.37286
O           4.02528         0.11260        -0.06689
H           4.43662         0.23366        -0.93551
H           2.52489         0.82137         0.19137

