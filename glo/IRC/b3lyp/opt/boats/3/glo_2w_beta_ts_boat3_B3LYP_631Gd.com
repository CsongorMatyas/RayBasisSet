%NPROCS=4
# opt=(MaxCycle=1000) freq B3LYP/6-31G(d)

 ./boats/3/glo_2w_beta_ts_boat3_B3LYP_631Gd.out

0  1
C           1.19518         0.19496         0.87209
C          -0.49082         1.75606        -0.21371
O          -0.14423        -0.74236        -0.95383
C          -1.02757         0.73130        -1.24550
C           1.17218        -0.47661        -0.52153
C          -0.02712         1.10883         1.09668
H          -0.63862         0.94727        -2.24358
H           0.29897         1.91259         1.77534
H          -1.32826         2.43151         0.00313
H           1.62741         0.24238        -1.22193
H           1.18710        -0.57804         1.65896
C           2.05085        -1.73651        -0.54345
H           1.99645        -2.22143        -1.52614
H           1.71560        -2.45473         0.21149
O          -2.31274         0.45684        -1.26470
O           3.41095        -1.40344        -0.20065
H           3.83649        -1.06052        -1.00241
O           0.56536         2.51839        -0.79002
H           1.38922         2.32104        -0.30777
O          -1.14119         0.41400         1.62652
H          -0.87875        -0.02418         2.45039
O           2.36391         1.01124         1.00238
H           3.10892         0.42072         0.77793
O          -1.47020        -2.28643         0.47876
H          -2.70819        -1.50335         0.58267
H          -0.90461        -1.66458        -0.08427
H          -1.61092        -3.07025        -0.07354
O          -3.44064        -0.80978         0.48365
H          -4.13353        -1.21712        -0.05833
H          -2.74553         0.08797        -0.39481

