%NPROCS=4
# opt=(MaxCycle=1000) freq B3LYP/6-31G(d)

 ./boats/5/glo_2w_alpha_ts_boat5_B3LYP_631Gd_r.out

0  1
O           0.33341        -0.67562        -1.05692
C          -0.82378        -0.11603         1.06867
C          -0.31489         1.76805        -0.64911
C          -1.23777         1.24263         0.48647
C           0.92059         0.90449        -0.98837
C          -0.55381        -1.15436        -0.02568
H          -1.19804         1.97812         1.29968
H          -0.04380        -2.01024         0.45489
H          -1.65706        -0.50622         1.67508
H          -0.94458         1.85105        -1.54045
C          -1.82466        -1.68043        -0.72373
H          -1.54035        -2.47169        -1.43466
H          -2.28820        -0.86944        -1.28656
O          -2.80656        -2.14374         0.20150
H          -2.50683        -2.99257         0.56244
O           0.18512         3.06219        -0.36081
H           1.03788         2.89527         0.08675
O          -2.55949         1.17566        -0.05796
H          -3.13999         0.76040         0.60467
O           0.30127         0.11764         1.90692
H           0.69998        -0.73636         2.15835
H           1.16682         0.96595        -2.06049
O           1.92992         1.05962        -0.19278
H           2.77417        -1.53666         0.10267
O           2.25891        -2.13044        -0.81077
H           1.99198        -3.01116        -0.51005
H           1.31082        -1.46372        -1.02443
O           3.09183        -0.71759         0.96398
H           4.03917        -0.52419         0.89026
H           2.55541         0.19298         0.51153

