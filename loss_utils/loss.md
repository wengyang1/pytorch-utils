# loss收敛问题
```commandline
1 loss无法收敛，可能的原因
lr大了，之前遇到过一个case，lr=0.001，一直无法收敛，loss反复横跳，改成lr=0.0001很快就收敛了
```
# 二分类
```commandline
二分类的loss中的inputs和targets的数据格式,
可以参考BCEWithLogitsLoss_demo.py
```