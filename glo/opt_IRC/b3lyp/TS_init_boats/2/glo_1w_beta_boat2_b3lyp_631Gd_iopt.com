%NPROCS=4
# opt=(calcfc,ts,noeigentest,maxcycle=1000,cartesian) b3lyp 6-31g(d) freq

 ./2/glo_1w_beta_boat2_b3lyp_631Gd_iopt.out

0  1
H          -1.14181        -1.17342        -1.71158
C          -1.17480        -0.60080        -0.76875
C           0.46599        -0.25690         1.16920
C           0.64249         0.95192        -1.07993
C           1.14508         0.91108         0.37534
O          -0.80172         0.75443        -1.04227
C          -0.23050        -1.28294         0.24987
H           1.26187        -0.78283         1.70983
H           1.07462         0.14710        -1.68546
H           0.79637         1.84132         0.84713
H          -0.81522        -1.96575         0.87665
C          -2.62841        -0.58590        -0.29278
H          -3.21547         0.05675        -0.96862
H          -2.67360        -0.16996         0.72263
O           0.90558         2.14381        -1.72907
H           0.96088         2.85392        -1.06671
O          -3.07868        -1.93561        -0.32855
H          -3.91057        -1.98787         0.17669
O           2.54615         0.95398         0.46484
H           2.92875         0.09107         0.17275
O          -0.46317         0.19014         2.14455
H          -0.89520         1.01804         1.83775
O           0.80145        -2.03001        -0.43010
H           0.37964        -2.70682        -0.98208
H          -2.65738         2.61392         1.02868
O          -1.70658         2.43967         0.95024
H          -1.57661         2.01597         0.07170
O           3.51600        -1.61667         0.00273
H           2.61885        -1.99920        -0.10683
H           3.79352        -1.86894         0.90091

