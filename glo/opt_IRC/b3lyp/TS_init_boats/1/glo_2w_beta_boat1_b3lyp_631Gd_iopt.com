%NPROCS=4
# opt=(calcfc,ts,noeigentest,maxcycle=1000,cartesian) b3lyp 6-31g(d) freq

 ./1/glo_2w_beta_boat1_b3lyp_631Gd_iopt.out

0  1
H          -0.59847        -2.26414         0.20245
C          -0.70639        -1.23190        -0.15735
C          -1.34555         1.08694         0.72664
C           0.93485         0.68537        -0.26941
C          -0.34327         1.50974        -0.34995
O           0.63252        -0.75349        -0.30525
C          -1.49216        -0.44263         0.89672
H           1.42155         0.86033         0.70108
H          -0.78435         1.35485        -1.34515
H          -2.55280        -0.71431         0.80441
C          -1.43128        -1.35239        -1.50550
H          -0.71334        -1.73874        -2.24245
H          -1.79771        -0.38016        -1.86010
O           1.76460         0.98836        -1.31444
H           2.68478         0.73273        -1.04808
H          -0.96882         1.47201         1.68372
O          -2.51398        -2.25479        -1.28692
H          -3.06136        -2.26691        -2.08586
O          -0.00167         2.87315        -0.17579
H          -0.84854         3.34934        -0.16193
O          -2.56894         1.73967         0.37306
H          -3.14872         1.73327         1.15047
O          -1.00123        -0.73659         2.21115
H          -1.14333        -1.68306         2.37295
O           2.68753        -1.67235         1.25721
H           1.88169        -1.49650         0.71541
H           2.47541        -1.29694         2.12550
O           4.14118         0.05362        -0.31067
H           3.76358        -0.59552         0.33464
H           4.57306        -0.49744        -0.98071

