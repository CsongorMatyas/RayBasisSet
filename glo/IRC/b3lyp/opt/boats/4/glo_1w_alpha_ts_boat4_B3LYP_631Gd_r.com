%NPROCS=4
# opt=(MaxCycle=1000) freq B3LYP/6-31G(d)

 ./boats/4/glo_1w_alpha_ts_boat4_B3LYP_631Gd_r.out

0  1
O          -1.24238         0.47615        -0.70049
C           1.04141         0.35192        -0.15891
C          -0.38803        -1.72785         0.45329
C           1.00878        -1.15820         0.18770
C          -1.51916        -1.34381        -0.49211
C          -0.27164         1.09227         0.13464
H           1.55474        -1.26751         1.13720
H          -0.53281         0.98315         1.19577
H           1.20888         0.42663        -1.24424
C          -0.23638         2.56317        -0.28607
H           0.00650         2.62610        -1.35050
H          -1.22521         3.02265        -0.14111
H          -0.29018        -2.82050         0.32748
O           0.77623         3.30683         0.39643
H           0.52335         3.37456         1.33146
O          -0.78330        -1.40119         1.77374
H          -1.75957        -1.48189         1.73714
O           1.58892        -1.93699        -0.83957
H           2.53255        -1.66070        -0.89860
O           2.13667         0.93695         0.55915
H           2.06669         1.90699         0.44630
H          -1.33355        -1.50445        -1.56230
O          -2.67855        -1.46572        -0.00905
O          -3.67299         0.71568        -0.55067
H          -3.58472        -0.27725        -0.41510
H          -2.20287         0.82043        -0.57578
H          -4.05814         1.04860         0.27464
O           4.08296        -0.71925        -0.53121
H           4.54349        -1.17265         0.19106
H           3.60285         0.01859        -0.09255

