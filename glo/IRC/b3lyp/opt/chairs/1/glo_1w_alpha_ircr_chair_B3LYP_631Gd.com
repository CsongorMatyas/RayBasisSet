%NPROCS=4
# opt=(MaxCycle=1000) freq B3LYP/6-31G(d)

 ./chairs/1/glo_1w_alpha_ircr_chair_B3LYP_631Gd.out

0  1
C           1.23190        -1.07216        -0.95057
O           1.20898        -1.10352         0.50165
H           4.53952        -1.42291         0.06997
O           3.99339        -0.63837         0.22957
H           3.18960        -0.97211         0.67611
C          -0.21261        -1.11040        -1.44052
C          -0.99477         0.09479        -0.90807
C          -0.88218         0.14832         0.62154
C           0.58110         0.04316         1.09417
H          -0.66760        -2.01702        -1.00696
H          -1.46994        -0.67368         1.04367
H           1.11748         0.95347         0.78851
H          -0.54713         1.00748        -1.33288
O          -0.25072        -1.15712        -2.84864
O          -2.32341        -0.08431        -1.36508
O          -1.49097         1.34746         1.17216
H          -2.83114         0.74729        -1.22270
H          -1.17761        -0.96708        -3.07752
H          -0.94038         2.10238         0.89658
C           0.71380        -0.11520         2.61414
O           0.31515         1.04826         3.31550
H           1.76721        -0.28385         2.85949
H           0.15098        -1.00748         2.93461
H          -0.59888         1.23740         3.04261
H           1.75355        -2.00048        -1.21508
O           1.88308         0.05165        -1.43251
H           2.81389        -0.01784        -1.12209
O          -3.70201         2.10399        -0.37982
H          -4.57817         1.75365        -0.15687
H          -3.14025         1.89446         0.39511

