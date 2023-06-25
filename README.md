# tiny-xgboost

A tiny xgboost implementation in Python & numpy.

## Benchmarking

Benchmarking against xgboost shows we can get identical results in terms of test set
scores and individual trees grown, but tiny-xgboost can easily by 5-10x slower.

Results from running `benchmark.py`:

```
Data size = (20640, 8)
Time tiny-XGB: 3.095 seconds
Time XGB: 0.522 seconds
tiny-XGB test-rmse = 0.50141
XGB test-rmse = 0.50144
tiny-XGB tree 81:
[f0<5.48540] gain=0.98343,cover=11610
        [f0<5.29440] gain=1.50335,cover=9821
                [f0<5.24895] gain=1.93965,cover=9595
                        leaf=0.00096 cover=9534
                        leaf=-0.05229 cover=61
                [f0<5.29545] gain=2.00560,cover=226
                        leaf=0.19671 cover=5
                        leaf=0.02058 cover=221
        [f5<2.26140] gain=2.51980,cover=1789
                [f6<33.98500] gain=2.59310,cover=148
                        leaf=0.07591 cover=64
                        leaf=-0.00387 cover=84
                [f2<5.43002] gain=3.48747,cover=1641
                        leaf=-0.07538 cover=69
                        leaf=-0.00693 cover=1572

XGB tree 81:
0:[f0<5.4854002] yes=1,no=2,missing=1,gain=0.984428108,cover=11610
        1:[f0<5.29440022] yes=3,no=4,missing=3,gain=1.50435829,cover=9821
                3:[f0<5.24895] yes=7,no=8,missing=7,gain=1.94064927,cover=9595
                        7:leaf=0.000958797405,cover=9534
                        8:leaf=-0.0522896349,cover=61
                4:[f0<5.29545021] yes=9,no=10,missing=9,gain=2.00659657,cover=226
                        9:leaf=0.196708128,cover=5
                        10:leaf=0.0205782466,cover=221
        2:[f5<2.26140165] yes=5,no=6,missing=5,gain=2.52078581,cover=1789
                5:[f6<33.9850006] yes=11,no=12,missing=11,gain=2.59409904,cover=148
                        11:leaf=0.0759064406,cover=64
                        12:leaf=-0.00387001457,cover=84
                6:[f2<5.43002176] yes=13,no=14,missing=13,gain=3.48847008,cover=1641
                        13:leaf=-0.0753841102,cover=69
                        14:leaf=-0.0069281566,cover=1572
```

