%NPROCS=4
# opt=(calcfc,ts,noeigentest,maxcycle=1000,cartesian) b3lyp 6-31g(d) freq

 ./1/glo_2w_alpha_boat1_b3lyp_631Gd_iopt.out

0  1
H          -1.24027        -1.87959        -0.13769
C          -1.14180        -0.80632        -0.34300
C           0.04394         1.14620         0.85793
C           0.66937         0.50731        -1.50610
C           0.13173         1.62176        -0.58858
O          -0.09919        -0.72055        -1.31293
C          -0.82481        -0.11840         1.00341
H          -0.88241         1.88541        -0.92896
H          -1.77592         0.16749         1.46487
C          -2.49833        -0.39020        -0.93160
H          -2.59933        -0.87469        -1.91284
H          -2.56412         0.69668        -1.08014
H           1.05941         0.90702         1.18951
O          -3.48548        -0.84251        -0.00887
H          -4.35457        -0.55954        -0.32955
O           0.98346         2.74028        -0.69746
H           0.70138         3.34308         0.01186
O          -0.46241         2.25306         1.61166
H          -0.18883         2.13919         2.53459
O          -0.11482        -1.01689         1.88893
H          -0.76668        -1.60586         2.30059
H           0.52318         0.79549        -2.55101
O           2.01055         0.23620        -1.37685
H           2.26157         0.03804        -0.44105
O           2.78795        -0.87781         1.05597
H           2.63747        -1.72217         0.56900
H           2.08296        -0.90364         1.72459
O           1.48166        -2.91315        -0.27146
H           0.84188        -2.80083         0.45016
H           1.16983        -2.25721        -0.92823

