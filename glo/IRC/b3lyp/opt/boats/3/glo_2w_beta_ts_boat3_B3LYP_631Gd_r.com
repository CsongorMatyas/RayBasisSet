%NPROCS=4
# opt=(MaxCycle=1000) freq B3LYP/6-31G(d)

 ./boats/3/glo_2w_beta_ts_boat3_B3LYP_631Gd_r.out

0  1
C           1.19542         0.19355         0.87504
C          -0.49439         1.76344        -0.21245
O          -0.12177        -0.80211        -0.93982
C          -1.09603         0.81010        -1.28276
C           1.18070        -0.49360        -0.50772
C          -0.02849         1.10741         1.09431
H          -0.55679         0.89865        -2.24184
H           0.30058         1.91011         1.77346
H          -1.31895         2.44736         0.02809
H           1.57419         0.23312        -1.23080
H           1.18962        -0.57506         1.66640
C           2.06056        -1.73834        -0.54131
H           1.99937        -2.21874        -1.52593
H           1.72421        -2.45958         0.21113
O          -2.29066         0.48069        -1.28291
O           3.41515        -1.40355        -0.20021
H           3.84416        -1.06702        -1.00257
O           0.56634         2.51782        -0.79092
H           1.38767         2.33073        -0.30177
O          -1.14045         0.41277         1.62565
H          -0.88148        -0.01527         2.45597
O           2.36356         1.01011         0.99986
H           3.11123         0.41914         0.78523
O          -1.47424        -2.28306         0.48295
H          -2.35070        -1.74769         0.56766
H          -0.65632        -1.41067        -0.28688
H          -1.68098        -3.05642        -0.06364
O          -3.46273        -0.81406         0.49313
H          -4.16836        -1.15558        -0.07748
H          -3.02515        -0.10274        -0.06559

